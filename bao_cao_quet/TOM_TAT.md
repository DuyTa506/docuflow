# Kết quả quét ATTT DocuFlow (09/11/2026)

## Quét tĩnh (mã nguồn + thư viện)
| Công cụ | Kết quả |
|---|---|
| Bandit (toàn bộ mã) | 0 HIGH, 1 MEDIUM, 46 LOW. MEDIUM duy nhất nằm ở `scripts/pdf_render_regression.py` (dùng /tmp) - không có trong gói nộp. LOW: `try/except pass`, gọi `subprocess` dạng danh sách (không shell), `assert` |
| detect-secrets | Chỉ còn mật khẩu mặc định dạng ví dụ (`.env.example`, `alembic.ini`), chuỗi kiểm tra trong `config/settings.py` (bị chặn khi `DOCUFLOW_PROD=1`), và băm CSP. Không có khóa thật |
| pip-audit | Giảm từ 22 gói có lỗ hổng xuống 8: `docling-slim`, `pip`, `setuptools`, `filelock` (đã nâng), `PyPDF2`, `python-jose`, `ecdsa`, `accelerate` (3 gói sau chưa có bản vá) |

## Quét động (máy chạy ứng dụng thật, quét từ ngoài qua tường lửa)
| Công cụ | Kết quả |
|---|---|
| Nmap 65.535 cổng | Chỉ `8022/tcp` mở; 65.534 cổng còn lại bị lọc |
| ZAP API, có xác thực (60 đường dẫn, 210 URL, tấn công chủ động: SQLi, XSS, command injection, path traversal, SSRF...) | 0 FAIL, 1 WARN (kiểu nội dung `text/html` của trang SPA - không phải lỗi). Máy chủ phục vụ 8.516 request, 0 lỗi 5xx |
| ZAP giao diện (baseline) | 0 FAIL, 5 WARN, **toàn bộ thuộc frontend Angular** (xem dưới) |

## Còn lại (thuộc frontend `Fe-Library`, repo riêng)
- Angular 21.0.5 bị ZAP gắn cờ "thư viện JS có lỗ hổng" - cần nâng Angular rồi build lại.
- `bypassSecurityTrustHtml(` trong `main.js` - cần rà chỗ dùng.
- CSP phải cho `style-src 'unsafe-inline'` vì Angular dùng style nội tuyến.
- 2 mục thông tin: nội dung có thể cache, ứng dụng web hiện đại.

## Đã sửa trong đợt quét này
- `eval()` trên kết quả OCR (thực thi mã từ xa) -> `ast.literal_eval`
- SSRF qua webhook; brute-force đăng nhập; thiếu header bảo mật; CORS `*`
- Health công khai lộ thông tin; SSE tin token cũ; tài liệu người khác trả 403 (lộ sự tồn tại)
- Ký tự NUL (`%00`) trong URL/JSON gây lỗi 500 (do ZAP tìm ra)
- Đổi mật khẩu mặc định, bật `DOCUFLOW_PROD=1`, đóng mọi cổng trừ 8022, nâng thư viện có CVE
