using System.Data;
using System.Text.RegularExpressions;
using Microsoft.Data.SqlClient;

namespace LibraryApp;

/// <summary>
/// The library's own SQL Server database: its catalogue table dbo.TaiLieu, plus
/// the docuflow.* tables DocuFlow writes results into (created from the same
/// mssql_schema.sql handed to any library).
/// </summary>
public sealed class LibraryDb(Settings settings)
{
    // Same catalogue as the Python demo (library_system.py), so both clients behave alike.
    static readonly (string Ma, string NhanDe, string Lang, string File, string Tasks)[] Seed =
    [
        ("TL-2026-0001", "Novelty in the military domain", "en",
            "en_quan_su_novelty_military_domain.pdf", "ocr,digest,translate"),
        ("TL-2026-0002", "Giáo dục song ngữ cho học sinh Khmer", "vi",
            "vi_giao_duc_khmer_bilingual_education.pdf", "ocr,digest,translate"),
    ];

    const string LibraryDdl = """
        IF OBJECT_ID(N'dbo.TaiLieu', N'U') IS NULL
        CREATE TABLE dbo.TaiLieu (
            MaTaiLieu     NVARCHAR(20)  NOT NULL PRIMARY KEY,
            NhanDe        NVARCHAR(500) NOT NULL,
            NgonNgu       NVARCHAR(10)  NOT NULL,
            DuongDanTep   NVARCHAR(500) NOT NULL,
            YeuCau        NVARCHAR(100) NOT NULL,
            TrangThaiXuLy NVARCHAR(30)  NOT NULL DEFAULT N'CHO_GUI',
            DocuFlowJobId NVARCHAR(32)  NULL,
            CapNhatLuc    DATETIME2     NULL
        );
        GO
        CREATE OR ALTER VIEW dbo.vw_TaiLieu_DocuFlow AS
        SELECT t.MaTaiLieu, t.NhanDe, t.TrangThaiXuLy,
               j.JobId, j.Status, j.OcrStatus, j.DigestStatus, j.TranslateStatus,
               r.Title, r.Authors, r.PublishYear, r.Keywords,
               LEN(r.OcrText) AS OcrChars, LEN(r.TranslationText) AS TranslationChars,
               (SELECT COUNT(*) FROM docuflow.DocuFlowChapter c WHERE c.JobId = j.JobId) AS Chapters
        FROM dbo.TaiLieu t
        LEFT JOIN docuflow.DocuFlowJob j ON j.JobId = t.DocuFlowJobId
        LEFT JOIN docuflow.DocuFlowResult r ON r.JobId = t.DocuFlowJobId;
        """;

    public void EnsureSchema()
    {
        using (var master = Open(settings.SqlMaster))
            Exec(master, "IF DB_ID(N'LibraryDB') IS NULL CREATE DATABASE LibraryDB");

        using var db = Open(settings.SqlLibrary);
        RunBatches(db, File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "mssql_schema.sql")));
        RunBatches(db, LibraryDdl);
        foreach (var s in Seed)
        {
            using var cmd = new SqlCommand("""
                IF NOT EXISTS (SELECT 1 FROM dbo.TaiLieu WHERE MaTaiLieu = @ma)
                INSERT dbo.TaiLieu (MaTaiLieu, NhanDe, NgonNgu, DuongDanTep, YeuCau)
                VALUES (@ma, @nhande, @lang, @file, @tasks)
                """, db);
            cmd.Parameters.AddWithValue("@ma", s.Ma);
            cmd.Parameters.AddWithValue("@nhande", s.NhanDe);
            cmd.Parameters.AddWithValue("@lang", s.Lang);
            cmd.Parameters.AddWithValue("@file", Path.Combine(settings.FilesDir, s.File));
            cmd.Parameters.AddWithValue("@tasks", s.Tasks);
            cmd.ExecuteNonQuery();
        }
    }

    public DataTable Catalogue() => Query("""
        SELECT MaTaiLieu AS [Mã tài liệu], NhanDe AS [Nhan đề], NgonNgu AS [Ngôn ngữ],
               YeuCau AS [Yêu cầu], TrangThaiXuLy AS [Trạng thái], DocuFlowJobId AS [Job DocuFlow],
               OcrStatus AS OCR, DigestStatus AS [Tổng thuật], TranslateStatus AS [Dịch],
               CapNhatLuc AS [Cập nhật]
        FROM dbo.TaiLieu t LEFT JOIN docuflow.DocuFlowJob j ON j.JobId = t.DocuFlowJobId
        ORDER BY MaTaiLieu
        """);

    /// <summary>What DocuFlow wrote into this database, read with plain SQL.</summary>
    public DataTable Results() => Query("""
        SELECT r.ExternalId AS [Mã tài liệu], r.Title AS [Nhan đề (DocuFlow)], r.Authors AS [Tác giả],
               r.PublishYear AS [Năm], r.Pages AS [Số trang],
               (SELECT COUNT(*) FROM docuflow.DocuFlowChapter c WHERE c.JobId = r.JobId) AS [Số chương],
               (SELECT COUNT(*) FROM docuflow.DocuFlowKeyword k WHERE k.JobId = r.JobId) AS [Số từ khoá],
               LEN(r.OcrText) AS [Ký tự OCR], LEN(r.TranslationText) AS [Ký tự bản dịch],
               LEN(r.MarcXml) AS [MARCXML], r.Abstract AS [Tóm tắt]
        FROM docuflow.DocuFlowResult r ORDER BY r.ExternalId
        """);

    public List<(string Ma, string Lang, string File, string Tasks)> Pending()
    {
        var rows = new List<(string, string, string, string)>();
        using var db = Open(settings.SqlLibrary);
        using var cmd = new SqlCommand(
            "SELECT MaTaiLieu, NgonNgu, DuongDanTep, YeuCau FROM dbo.TaiLieu WHERE TrangThaiXuLy = N'CHO_GUI'", db);
        using var r = cmd.ExecuteReader();
        while (r.Read()) rows.Add((r.GetString(0), r.GetString(1), r.GetString(2), r.GetString(3)));
        return rows;
    }

    /// <summary>Jobs sent but not yet final — reconciled on start-up, after a restart.</summary>
    public List<string> InFlightJobs()
    {
        var jobs = new List<string>();
        using var db = Open(settings.SqlLibrary);
        using var cmd = new SqlCommand("""
            SELECT DocuFlowJobId FROM dbo.TaiLieu
            WHERE DocuFlowJobId IS NOT NULL AND TrangThaiXuLy NOT IN (N'HOAN_THANH', N'MOT_PHAN', N'LOI')
            """, db);
        using var r = cmd.ExecuteReader();
        while (r.Read()) jobs.Add(r.GetString(0));
        return jobs;
    }

    public void SetStatus(string ma, string status, string? jobId = null)
    {
        using var db = Open(settings.SqlLibrary);
        using var cmd = new SqlCommand("""
            UPDATE dbo.TaiLieu SET TrangThaiXuLy = @s, DocuFlowJobId = COALESCE(@job, DocuFlowJobId),
                   CapNhatLuc = SYSDATETIME()
            WHERE MaTaiLieu = @ma
            """, db);
        cmd.Parameters.AddWithValue("@s", status);
        cmd.Parameters.AddWithValue("@job", (object?)jobId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("@ma", ma);
        cmd.ExecuteNonQuery();
    }

    public void Reset()
    {
        using var master = Open(settings.SqlMaster);
        Exec(master, """
            IF DB_ID(N'LibraryDB') IS NOT NULL
            BEGIN ALTER DATABASE LibraryDB SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE LibraryDB; END
            """);
    }

    public string ServerInfo()
    {
        using var db = Open(settings.SqlMaster);
        using var cmd = new SqlCommand(
            "SELECT LEFT(@@VERSION, CHARINDEX(CHAR(10), @@VERSION) - 1) + ' · ' + CAST(SERVERPROPERTY('Edition') AS nvarchar)", db);
        return $"{cmd.ExecuteScalar()?.ToString()?.Trim()} @ {db.DataSource}";
    }

    DataTable Query(string sql)
    {
        using var db = Open(settings.SqlLibrary);
        using var da = new SqlDataAdapter(sql, db);
        var t = new DataTable();
        da.Fill(t);
        return t;
    }

    static SqlConnection Open(string cs)
    {
        var c = new SqlConnection(cs);
        c.Open();
        return c;
    }

    static void Exec(SqlConnection c, string sql)
    {
        using var cmd = new SqlCommand(sql, c);
        cmd.ExecuteNonQuery();
    }

    static void RunBatches(SqlConnection c, string sql)
    {
        foreach (var batch in Regex.Split(sql, @"^\s*GO\s*$", RegexOptions.Multiline | RegexOptions.IgnoreCase))
            if (!string.IsNullOrWhiteSpace(batch))
                Exec(c, batch);
    }
}
