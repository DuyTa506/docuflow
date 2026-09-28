using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace LibraryApp;

/// <summary>The DocuFlow integration API, called over the LAN with the library's API key.</summary>
public sealed class DocuFlowClient
{
    readonly HttpClient _http;
    readonly Settings _settings;

    public DocuFlowClient(Settings settings)
    {
        _settings = settings;
        _http = new HttpClient { BaseAddress = new Uri(settings.DocuFlowUrl.TrimEnd('/') + "/"),
                                 Timeout = TimeSpan.FromMinutes(5) };
        _http.DefaultRequestHeaders.Add("X-API-Key", settings.ApiKey);
    }

    public async Task<JsonElement> PingAsync() => await GetJsonAsync("ping");

    public async Task<JsonElement> SubmitAsync(string externalId, string file, string lang, string tasks)
    {
        using var form = new MultipartFormDataContent();
        var bytes = new ByteArrayContent(await File.ReadAllBytesAsync(file));
        bytes.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
        form.Add(bytes, "file", Path.GetFileName(file));
        form.Add(new StringContent(externalId), "external_id");
        form.Add(new StringContent(lang), "source_language");
        form.Add(new StringContent(tasks), "tasks");
        form.Add(new StringContent(_settings.CallbackUrl), "callback_url");
        using var resp = await _http.PostAsync("jobs", form);
        return await ReadAsync(resp);
    }

    public async Task<JsonElement> JobAsync(string jobId) => await GetJsonAsync($"jobs/{jobId}");

    public async Task<JsonElement> GetJsonAsync(string pathOrUrl)
    {
        using var resp = await _http.GetAsync(pathOrUrl);
        return await ReadAsync(resp);
    }

    static async Task<JsonElement> ReadAsync(HttpResponseMessage resp)
    {
        var body = await resp.Content.ReadAsStringAsync();
        if (!resp.IsSuccessStatusCode)
            throw new HttpRequestException($"HTTP {(int)resp.StatusCode}: {body}", null, resp.StatusCode);
        return JsonDocument.Parse(body).RootElement.Clone();
    }

    /// <summary>
    /// X-DocuFlow-Signature = "sha256=" + hex(HMAC-SHA256(secret, "&lt;timestamp&gt;." + body)),
    /// rejected when the timestamp is more than 5 minutes off.
    /// </summary>
    public static bool Verify(string secret, string? timestamp, byte[] body, string? signature)
    {
        if (!long.TryParse(timestamp, out var ts) ||
            Math.Abs(DateTimeOffset.UtcNow.ToUnixTimeSeconds() - ts) > 300)
            return false;
        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(secret));
        var signed = Encoding.UTF8.GetBytes(timestamp + ".").Concat(body).ToArray();
        var expected = "sha256=" + Convert.ToHexString(hmac.ComputeHash(signed)).ToLowerInvariant();
        return CryptographicOperations.FixedTimeEquals(
            Encoding.ASCII.GetBytes(expected), Encoding.ASCII.GetBytes(signature ?? ""));
    }
}

/// <summary>Receives DocuFlow's signed callbacks on the library workstation.</summary>
public sealed class CallbackServer(string prefix, string secret,
                                   Action<IPEndPoint, JsonElement> onEvent, Action<string> log)
{
    readonly HttpListener _listener = new();

    public void Start()
    {
        _listener.Prefixes.Add(prefix);
        _listener.Start();
        _ = Task.Run(LoopAsync);
    }

    async Task LoopAsync()
    {
        while (_listener.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = await _listener.GetContextAsync(); }
            catch (Exception) when (!_listener.IsListening) { return; }

            // One bad request must never end the loop: the port would stay closed until restart.
            try { await HandleAsync(ctx); }
            catch (Exception e) { log($"Lỗi xử lý callback: {e.Message}"); }
        }
    }

    async Task HandleAsync(HttpListenerContext ctx)
    {
        // Read everything from the request first: closing the response disposes it.
        var req = ctx.Request;
        var from = req.RemoteEndPoint;
        var evt = req.Headers["X-DocuFlow-Event"];
        using var ms = new MemoryStream();
        await req.InputStream.CopyToAsync(ms);
        var body = ms.ToArray();
        var ok = DocuFlowClient.Verify(secret, req.Headers["X-DocuFlow-Timestamp"], body,
                                       req.Headers["X-DocuFlow-Signature"]);
        ctx.Response.StatusCode = ok ? 204 : 401;
        ctx.Response.Close();
        if (!ok)
        {
            log($"TỪ CHỐI callback từ {from}: chữ ký không hợp lệ");
            return;
        }
        log($"Callback {evt} từ {from.Address} — chữ ký HMAC hợp lệ");
        onEvent(from, JsonDocument.Parse(body).RootElement.Clone());
    }
}
