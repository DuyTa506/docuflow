using System.Text.Json;

namespace LibraryApp;

/// <summary>appsettings.json next to the exe (copy appsettings.example.json and fill in the keys).</summary>
public sealed record Settings(
    string DocuFlowUrl,
    string ApiKey,
    string WebhookSecret,
    string CallbackListen,
    string CallbackUrl,
    string SqlMaster,
    string FilesDir)
{
    public string SqlLibrary =>
        new Microsoft.Data.SqlClient.SqlConnectionStringBuilder(SqlMaster) { InitialCatalog = "LibraryDB" }
            .ConnectionString;

    public static Settings Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<Settings>(json)
               ?? throw new InvalidOperationException($"Cannot read {path}");
    }
}
