# Demo: DocuFlow ↔ hệ thống quản lý thư viện (MSSQL)

Chạy thử đầu cuối API tích hợp `/api/v2/integration` (báo cáo 5.4).

| Thành phần | Vai trò |
|---|---|
| `docker-compose.yml` | SQL Server 2022 (cổng 14330) đóng vai CSDL của thư viện |
| `library_system.py` | Phần mềm thư viện giả lập: bảng `dbo.TaiLieu`, gửi tài liệu, nhận callback, đọc kết quả. Không dùng mã DocuFlow — dùng làm mã mẫu cho bên thư viện |
| `demo_worker.py` | Worker chỉ phục vụ hàng đợi tích hợp của demo, ghi vào `LibraryDB` |
| `start_demo.sh` | Chạy tất cả |

## Yêu cầu

Hệ thống DocuFlow đang chạy (Postgres, MinIO, Temporal, worker extraction và pipeline, LLM),
Docker, `.venv` có `pymssql` (`uv pip install pymssql`).

## Chạy

```bash
bash deploy/integration/demo/start_demo.sh
```

Script sẽ:

1. bật SQL Server, cấp khoá API cho "Thư viện Demo" (lưu ở `.state/client.json`, đã gitignore);
2. chạy một API DocuFlow phụ ở `:8023` và `demo_worker.py` trên hàng đợi
   `docuflow-integration-demo` — không động tới API và worker đang phục vụ người dùng;
   OCR, tổng thuật và dịch vẫn do các worker DocuFlow sẵn có làm;
3. `library_system.py run`: tạo `LibraryDB` + schema `docuflow`, gửi hai tài liệu trong
   `dbo.TaiLieu`, chờ callback, kiểm chữ ký, lấy kết quả, in bảng kết quả trong MSSQL.

Chạy lại từ đầu: `python deploy/integration/demo/library_system.py reset` rồi chạy lại script.

Xem kết quả bằng SQL:

```bash
docker exec -it docuflow-library-mssql /opt/mssql-tools18/bin/sqlcmd -C -S localhost -U sa \
  -P 'DocuFlow#Demo2026' -d LibraryDB -Q "SELECT * FROM dbo.vw_TaiLieu_DocuFlow"
```

Kiểm thử bộ ghi với SQL Server thật:

```bash
LIBRARY_MSSQL_TEST_URL='mssql+pymssql://sa:DocuFlow#Demo2026@localhost:14330/master' \
  pytest tests/integration/test_mssql_sink_live.py --override-ini="addopts="
```

Mật khẩu `sa` ở đây chỉ dùng cho demo (đổi bằng biến `LIBRARY_SA_PASSWORD`).
