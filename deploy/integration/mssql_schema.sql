-- DocuFlow → thư viện: bảng nhận kết quả trong CSDL MSSQL của hệ thống thư viện.
-- Chạy một lần trên CSDL thư viện (SQL Server 2016+). Idempotent: chạy lại không lỗi.
-- DocuFlow ghi bằng một tài khoản chỉ có quyền trên schema [docuflow] (xem cuối file).
--
--   DocuFlowJob      trạng thái từng công việc, cập nhật khi mỗi bước đổi trạng thái
--   DocuFlowResult   kết quả của công việc đã xong: trường biên mục, văn bản, MARCXML
--   DocuFlowChapter  các chương của tổng thuật
--   DocuFlowKeyword  các từ khoá của tổng thuật

IF SCHEMA_ID(N'docuflow') IS NULL EXEC(N'CREATE SCHEMA [docuflow]');
GO

IF OBJECT_ID(N'docuflow.DocuFlowJob', N'U') IS NULL
CREATE TABLE [docuflow].[DocuFlowJob] (
    [JobId]           NVARCHAR(32)   NOT NULL PRIMARY KEY,
    [ExternalId]      NVARCHAR(100)  NULL,      -- mã biểu ghi phía thư viện
    [DocumentId]      NVARCHAR(32)   NULL,      -- mã tài liệu trong DocuFlow
    [Status]          NVARCHAR(20)   NOT NULL,  -- QUEUED | PROCESSING | COMPLETED | PARTIAL | FAILED
    [OcrStatus]       NVARCHAR(20)   NULL,      -- PENDING | RUNNING | COMPLETED | FAILED | SKIPPED
    [DigestStatus]    NVARCHAR(20)   NULL,
    [TranslateStatus] NVARCHAR(20)   NULL,
    [Error]           NVARCHAR(MAX)  NULL,
    [ResultUrl]       NVARCHAR(500)  NULL,
    [CreatedAt]       DATETIME2      NULL,
    [UpdatedAt]       DATETIME2      NULL,
    [CompletedAt]     DATETIME2      NULL,
    [SyncedAt]        DATETIME2      NOT NULL
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_DocuFlowJob_ExternalId')
    CREATE INDEX [IX_DocuFlowJob_ExternalId] ON [docuflow].[DocuFlowJob] ([ExternalId]);
GO

IF OBJECT_ID(N'docuflow.DocuFlowResult', N'U') IS NULL
CREATE TABLE [docuflow].[DocuFlowResult] (
    [JobId]              NVARCHAR(32)   NOT NULL PRIMARY KEY
                         REFERENCES [docuflow].[DocuFlowJob] ([JobId]),
    [ExternalId]         NVARCHAR(100)  NULL,
    [DocumentId]         NVARCHAR(32)   NULL,
    [Title]              NVARCHAR(1000) NULL,
    [Authors]            NVARCHAR(1000) NULL,
    [Publisher]          NVARCHAR(500)  NULL,
    [PublishYear]        NVARCHAR(20)   NULL,
    [ISBN]               NVARCHAR(50)   NULL,
    [DOI]                NVARCHAR(200)  NULL,
    [Pages]              NVARCHAR(20)   NULL,
    [SourceLanguage]     NVARCHAR(10)   NULL,
    [Abstract]           NVARCHAR(MAX)  NULL,   -- tóm tắt tài liệu (tiếng Việt)
    [Keywords]           NVARCHAR(MAX)  NULL,   -- nối bằng "; "; chi tiết ở DocuFlowKeyword
    [UsageScope]         NVARCHAR(MAX)  NULL,   -- JSON
    [ResearchDirections] NVARCHAR(MAX)  NULL,   -- JSON
    [MarcXml]            NVARCHAR(MAX)  NULL,   -- MARC21 slim
    [DublinCore]         NVARCHAR(MAX)  NULL,   -- oai_dc
    [DigestJson]         NVARCHAR(MAX)  NULL,   -- toàn bộ tổng thuật, JSON (chương: DocuFlowChapter)
    [OcrText]            NVARCHAR(MAX)  NULL,   -- văn bản số hoá (markdown)
    [TranslationText]    NVARCHAR(MAX)  NULL,   -- bản dịch tiếng Việt
    [SyncedAt]           DATETIME2      NOT NULL
);
GO

IF OBJECT_ID(N'docuflow.DocuFlowChapter', N'U') IS NULL
CREATE TABLE [docuflow].[DocuFlowChapter] (
    [JobId]         NVARCHAR(32)   NOT NULL REFERENCES [docuflow].[DocuFlowJob] ([JobId]),
    [Ord]           INT            NOT NULL,
    [ChapterNo]     INT            NULL,
    [TitleVi]       NVARCHAR(1000) NULL,
    [TitleOriginal] NVARCHAR(1000) NULL,
    [Content]       NVARCHAR(MAX)  NULL,
    PRIMARY KEY ([JobId], [Ord])
);
GO

IF OBJECT_ID(N'docuflow.DocuFlowKeyword', N'U') IS NULL
CREATE TABLE [docuflow].[DocuFlowKeyword] (
    [JobId]   NVARCHAR(32)  NOT NULL REFERENCES [docuflow].[DocuFlowJob] ([JobId]),
    [Ord]     INT           NOT NULL,
    [Keyword] NVARCHAR(500) NULL,
    [Display] NVARCHAR(500) NULL,
    [Weight]  FLOAT         NULL,
    PRIMARY KEY ([JobId], [Ord])
);
GO

-- Tài khoản DocuFlow dùng để ghi (đổi mật khẩu trước khi dùng thật):
-- CREATE LOGIN docuflow_writer WITH PASSWORD = N'<mật khẩu mạnh>';
-- CREATE USER docuflow_writer FOR LOGIN docuflow_writer WITH DEFAULT_SCHEMA = [docuflow];
-- GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA::[docuflow] TO docuflow_writer;
