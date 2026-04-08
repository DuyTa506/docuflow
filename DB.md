// Use DBML to define your database structure
// Docs: https://dbml.dbdiagram.io/docs

// 0. ID GENERATOR
Table id_sequences {
  table_name varchar [primary key, note: 'e.g., documents, users']
  prefix varchar [not null, note: 'e.g., DOC_, USR_']
  current_value integer [default: 0, note: 'Last issued ID number']
  increment_by integer [default: 1]
}

// 1. USERS & ROLES
Table users {
  id varchar [primary key, note: 'e.g., USR_001']
  username varchar [unique, not null]
  password_hash varchar [not null]
  full_name varchar [not null]
  email varchar [unique]
  
  // Phân quyền: TEACHER, LIBRARIAN, ADMIN
  role varchar [not null, default: 'TEACHER', note: 'TEACHER: Giáo viên, LIBRARIAN: NV Thư viện, ADMIN: Quản trị viên']
  
  // Trạng thái: ACTIVE, PENDING_APPROVAL, INACTIVE
  status varchar [not null, default: 'ACTIVE', note: 'PENDING_APPROVAL dùng khi LIBRARIAN đăng ký, chờ Admin duyệt']
  
  created_at timestamp [default: `now()`]
}

// 2. DIGITIZED DOCUMENTS
Table documents {
  id varchar [primary key, note: 'e.g., DOC_001']
  user_id varchar [not null, note: 'Người upload/sở hữu tài liệu']
  title varchar [not null, note: 'Tên tài liệu hiển thị']
  source_language varchar [not null, note: 'vi, en, ru, zh - Để rẽ nhánh model OCR/Dịch']
  
  // Thông tin file
  original_file_name varchar [not null]
  format varchar [not null, note: 'pdf, docx, image']
  file_path text [not null, note: 'Đường dẫn lưu trên ổ cứng/MinIO']
  page_count integer [note: 'Dùng để làm progress bar OCR (tùy chọn)']
  
  // Trạng thái & Audit
  processing_status varchar [not null, note: 'INIT, EXTRACT_IN_PROGRESS, EXTRACTED, FAILED']
  is_synced_to_library boolean [default: false, note: 'Đánh dấu tài liệu đã được NV Thư viện đồng bộ sang kho chung chưa']
  created_at timestamp [default: `now()`]
}

Table digitized_texts {
  id varchar [primary key, note: 'e.g., txt_DOC_001']
  document_id varchar [not null]
  ocr_content text [note: 'Raw text from PaddleOCR/VinternOCR']
  normalized_content text [note: 'Cleaned text after NLP pipeline']
}

// 3. INFORMATION EXTRACTION
Table translations {
  id varchar [primary key, note: 'e.g., trans_en_vi_DOC_001']
  document_id varchar [not null]
  target_language varchar [default: 'vi']
  translated_content text [not null]
  status varchar [note: 'PENDING_REVIEW, APPROVED']
}

Table summaries {
  id varchar [primary key, note: 'e.g., sum_short_DOC_001']
  document_id varchar [not null]
  summary_type varchar [not null, note: 'short, detailed']
  content text [not null]
}

Table main_contents {
  id varchar [primary key, note: 'e.g., main_DOC_001']
  document_id varchar [not null]
  details jsonb [not null, note: 'Store JSON of main points, methods, results']
}

// 4. KEYWORDS & RESEARCH DIRECTIONS (N:N)
Table keywords {
  id varchar [primary key, note: 'e.g., kw_nlp, kw_ocr']
  keyword_name varchar [unique, not null]
}

Table document_keywords {
  document_id varchar
  keyword_id varchar
  weight float [note: 'TF-IDF or TextRank score']

  indexes {
    (document_id, keyword_id) [pk]
  }
}

Table research_directions {
  id varchar [primary key, note: 'e.g., rs_ai']
  direction_name varchar [unique, not null]
  is_predefined boolean [default: true, note: 'True if from catalog, False if LLM suggested']
}

Table document_research_directions {
  document_id varchar
  direction_id varchar

  indexes {
    (document_id, direction_id) [pk]
  }
}

// RELATIONSHIPS
Ref: documents.user_id > users.id

Ref: digitized_texts.document_id - documents.id
Ref: translations.document_id > documents.id
Ref: summaries.document_id > documents.id
Ref: main_contents.document_id - documents.id

Ref: document_keywords.document_id > documents.id
Ref: document_keywords.keyword_id > keywords.id

Ref: document_research_directions.document_id > documents.id
Ref: document_research_directions.direction_id > research_directions.id