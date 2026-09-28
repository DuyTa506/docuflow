using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace LibraryApp;

public sealed class MainForm : Form
{
    readonly Settings _settings;
    readonly LibraryDb _db;
    readonly DocuFlowClient _docuflow;
    CallbackServer? _callbacks;  // held for the app's lifetime so the listener is never collected

    readonly Label _info = new() { Dock = DockStyle.Top, AutoSize = false, Height = 64, Padding = new Padding(8, 6, 8, 0),
                                   Font = new Font("Segoe UI", 9.5f), BackColor = Color.FromArgb(232, 240, 250) };
    readonly DataGridView _catalogue = Grid();
    readonly DataGridView _results = Grid();
    readonly TextBox _abstract = new() { Dock = DockStyle.Bottom, Height = 90, Multiline = true, ReadOnly = true,
                                         ScrollBars = ScrollBars.Vertical, Font = new Font("Segoe UI", 9.5f) };
    readonly TextBox _log = new() { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical,
                                    Font = new Font("Consolas", 9.5f), BackColor = Color.FromArgb(24, 24, 24),
                                    ForeColor = Color.Gainsboro, WordWrap = false };

    public MainForm(Settings settings)
    {
        _settings = settings;
        _db = new LibraryDb(settings);
        _docuflow = new DocuFlowClient(settings);

        Text = "Phần mềm Quản lý Thư viện (mô phỏng) — tích hợp DocuFlow";
        Width = 1280; Height = 800;
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Segoe UI", 9.5f);

        var title = new Label { Dock = DockStyle.Top, Height = 34, Text = "  PHẦN MỀM QUẢN LÝ THƯ VIỆN  ·  C# .NET 8 WinForms trên Windows",
                                Font = new Font("Segoe UI Semibold", 13f), ForeColor = Color.White,
                                BackColor = Color.FromArgb(0, 84, 147), TextAlign = ContentAlignment.MiddleLeft };

        var bar = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 42, Padding = new Padding(6, 6, 6, 0) };
        bar.Controls.AddRange([
            Button("Kiểm tra kết nối DocuFlow", async () => await PingAsync()),
            Button("Gửi tài liệu sang DocuFlow", async () => await SubmitAsync()),
            Button("Cập nhật trạng thái (GET /jobs)", async () => await ReconcileAsync()),
            Button("Làm mới", () => { Reload(); return Task.CompletedTask; }),
            Button("Xoá CSDL && làm lại", async () => { _db.Reset(); _db.EnsureSchema(); Reload();
                                                         Log("Đã tạo lại LibraryDB"); await Task.CompletedTask; }),
        ]);

        var tabs = new TabControl { Dock = DockStyle.Fill };
        var t1 = new TabPage("Danh mục tài liệu (dbo.TaiLieu)");
        t1.Controls.Add(_catalogue);
        var t2 = new TabPage("Kết quả DocuFlow ghi vào CSDL thư viện (docuflow.*)");
        t2.Controls.Add(_results);
        t2.Controls.Add(_abstract);
        tabs.TabPages.AddRange([t1, t2]);
        _results.SelectionChanged += (_, _) =>
            _abstract.Text = _results.CurrentRow?.Cells["Tóm tắt"].Value?.ToString() ?? "";

        var split = new SplitContainer { Dock = DockStyle.Fill, Orientation = Orientation.Horizontal };
        split.Panel1.Controls.Add(tabs);
        var logBox = new GroupBox { Dock = DockStyle.Fill, Text = "Nhật ký trao đổi với DocuFlow" };
        logBox.Controls.Add(_log);
        split.Panel2.Controls.Add(logBox);

        Controls.Add(split);
        Controls.Add(bar);
        Controls.Add(_info);
        Controls.Add(title);

        var timer = new System.Windows.Forms.Timer { Interval = 5000 };
        timer.Tick += (_, _) => Reload();
        timer.Start();
        Shown += async (_, _) => { split.SplitterDistance = 320; await StartAsync(); };
    }

    async Task StartAsync()
    {
        // The log survives restarts: show where the previous session left off.
        var logFile = Path.Combine(AppContext.BaseDirectory, "thuvien.log");
        if (File.Exists(logFile))
        {
            foreach (var line in File.ReadLines(logFile).TakeLast(40))
                _log.AppendText(line + Environment.NewLine);
            _log.AppendText("──────── khởi động lại phần mềm ────────" + Environment.NewLine);
        }

        string sql;
        try
        {
            _db.EnsureSchema();
            sql = _db.ServerInfo();
            Log($"CSDL thư viện sẵn sàng: LibraryDB (dbo.TaiLieu + schema docuflow) — {sql}");
        }
        catch (Exception e) { sql = "lỗi: " + e.Message; Log("Không kết nối được SQL Server: " + e.Message); }

        _info.Text = $"Máy: {Environment.MachineName}   ·   {OsName()}   ·   IP: {LocalIps()}   ·   .NET {Environment.Version}\n" +
                     $"DocuFlow: {_settings.DocuFlowUrl}   ·   Callback: {_settings.CallbackUrl}\n" +
                     $"CSDL thư viện: {sql}";

        try
        {
            _callbacks = new CallbackServer(_settings.CallbackListen, _settings.WebhookSecret, OnCallback, Log);
            _callbacks.Start();
            Log($"Đang nghe callback tại {_settings.CallbackListen}");
        }
        catch (Exception e) { Log("Không mở được cổng callback: " + e.Message); }

        Reload();
        await PingAsync();
        // After a restart of this app, catch up on anything that finished meanwhile.
        if (_db.InFlightJobs().Count > 0) await ReconcileAsync();
    }

    async Task PingAsync()
    {
        try
        {
            var p = await _docuflow.PingAsync();
            Log($"GET {_settings.DocuFlowUrl}/ping → 200: {p}");
        }
        catch (Exception e) { Log("Không kết nối được DocuFlow: " + e.Message); }
    }

    async Task SubmitAsync()
    {
        var pending = _db.Pending();
        if (pending.Count == 0) { Log("Không có tài liệu nào ở trạng thái CHO_GUI"); return; }
        foreach (var (ma, lang, file, tasks) in pending)
        {
            try
            {
                Log($"POST /jobs  {ma}  ({Path.GetFileName(file)}, {new FileInfo(file).Length / 1024} KB, tasks={tasks})");
                var job = await _docuflow.SubmitAsync(ma, file, lang, tasks);
                var id = job.GetProperty("job_id").GetString()!;
                _db.SetStatus(ma, "DA_GUI", id);
                Log($"  → 202 {id}, trạng thái {job.GetProperty("status").GetString()}");
            }
            catch (Exception e) { Log($"  → lỗi gửi {ma}: {e.Message}"); }
        }
        Reload();
    }

    async Task ReconcileAsync()
    {
        foreach (var id in _db.InFlightJobs())
        {
            try
            {
                var job = await _docuflow.JobAsync(id);
                Log($"GET /jobs/{id} → {job.GetProperty("status").GetString()} {job.GetProperty("stages")}");
                await ApplyAsync(job);
            }
            catch (Exception e) { Log($"GET /jobs/{id} lỗi: {e.Message}"); }
        }
        Reload();
    }

    void OnCallback(IPEndPoint from, JsonElement payload) =>
        BeginInvoke(async () =>
        {
            var job = payload.GetProperty("job");
            Log($"  {job.GetProperty("external_id").GetString()} ({job.GetProperty("job_id").GetString()}): {job.GetProperty("stages")}");
            await ApplyAsync(job);
            Reload();
        });

    async Task ApplyAsync(JsonElement job)
    {
        var status = job.GetProperty("status").GetString();
        var ma = job.GetProperty("external_id").GetString()!;
        var mapped = status switch
        {
            "COMPLETED" => "HOAN_THANH", "PARTIAL" => "MOT_PHAN", "FAILED" => "LOI",
            "PROCESSING" => "DANG_XU_LY", _ => "DA_GUI",
        };
        _db.SetStatus(ma, mapped);
        if (status is not ("COMPLETED" or "PARTIAL")) return;

        var result = await _docuflow.GetJsonAsync(job.GetProperty("links").GetProperty("result").GetString()!);
        string Len(string part, string field) =>
            result.TryGetProperty(part, out var p) && p.ValueKind == JsonValueKind.Object &&
            p.TryGetProperty(field, out var f) && f.ValueKind == JsonValueKind.String ? $"{f.GetString()!.Length}" : "0";
        var digest = result.GetProperty("digest");
        int Count(string field) => digest.ValueKind == JsonValueKind.Object &&
                                   digest.TryGetProperty(field, out var a) && a.ValueKind == JsonValueKind.Array
                                   ? a.GetArrayLength() : 0;
        Log($"  GET {job.GetProperty("links").GetProperty("result").GetString()} → OCR {Len("ocr", "text")} ký tự, " +
            $"{Count("chapters")} chương, {Count("keywords")} từ khoá, bản dịch {Len("translation", "text")} ký tự");
    }

    void Reload()
    {
        try
        {
            Keep(_catalogue, _db.Catalogue());
            Keep(_results, _db.Results());
            if (_results.Columns.Contains("Tóm tắt")) _results.Columns["Tóm tắt"]!.Visible = false;
        }
        catch (Exception) { /* SQL Server not up yet: the timer retries */ }
    }

    static void Keep(DataGridView g, System.Data.DataTable t)
    {
        var row = g.CurrentCell?.RowIndex ?? 0;
        g.DataSource = t;
        foreach (DataGridViewColumn c in g.Columns)
            if (c.Width > 300) { c.AutoSizeMode = DataGridViewAutoSizeColumnMode.None; c.Width = 300; }
        if (row < g.Rows.Count && g.Columns.Count > 0) g.CurrentCell = g.Rows[row].Cells[0];
    }

    void Log(string msg)
    {
        if (InvokeRequired) { BeginInvoke(() => Log(msg)); return; }
        _log.AppendText($"[{DateTime.Now:HH:mm:ss}] {msg}{Environment.NewLine}");
        try { File.AppendAllText(Path.Combine(AppContext.BaseDirectory, "thuvien.log"),
                                 $"[{DateTime.Now:HH:mm:ss}] {msg}{Environment.NewLine}"); }
        catch (IOException) { }
    }

    /// <summary>"Microsoft Windows 11 Enterprise LTSC …" rather than the kernel version .NET reports.</summary>
    static string OsName()
    {
        try
        {
            using var q = new System.Management.ManagementObjectSearcher("SELECT Caption, Version FROM Win32_OperatingSystem");
            foreach (var o in q.Get()) return $"{o["Caption"]} ({o["Version"]})";
        }
        catch (Exception) { }
        return RuntimeInformation.OSDescription;
    }

    static string LocalIps() => string.Join(", ",
        NetworkInterface.GetAllNetworkInterfaces()
            .Where(n => n.OperationalStatus == OperationalStatus.Up && n.NetworkInterfaceType != NetworkInterfaceType.Loopback)
            .SelectMany(n => n.GetIPProperties().UnicastAddresses)
            .Where(a => a.Address.AddressFamily == AddressFamily.InterNetwork)
            .Select(a => a.Address.ToString()));

    static DataGridView Grid() => new()
    {
        Dock = DockStyle.Fill, ReadOnly = true, AllowUserToAddRows = false, RowHeadersVisible = false,
        AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.AllCells, BackgroundColor = Color.White,
        SelectionMode = DataGridViewSelectionMode.FullRowSelect,
    };

    static Button Button(string text, Func<Task> onClick)
    {
        var b = new Button { Text = text, AutoSize = true, Height = 30, Margin = new Padding(0, 0, 8, 0) };
        b.Click += async (_, _) => { b.Enabled = false; try { await onClick(); } finally { b.Enabled = true; } };
        return b;
    }
}
