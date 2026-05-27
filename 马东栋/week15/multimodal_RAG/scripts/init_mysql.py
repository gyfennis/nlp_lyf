import sys
sys.path.insert(0, ".")

from src.storage.mysql_client import MySQLClient

TABLES_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(500),
    file_type VARCHAR(20),
    file_path VARCHAR(1000) NOT NULL,
    file_size BIGINT,
    page_count INT,
    status ENUM('uploaded', 'processing', 'completed', 'failed'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    error_message TEXT,
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) NOT NULL,
    content TEXT NOT NULL,
    content_type ENUM('text', 'table', 'image', 'formula', 'mixed'),
    page_number INT,
    position INT,
    token_count INT,
    vector_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    INDEX idx_document_id (document_id),
    INDEX idx_content_type (content_type)
);

CREATE TABLE IF NOT EXISTS images (
    id VARCHAR(36) PRIMARY KEY,
    chunk_id VARCHAR(36),
    document_id VARCHAR(36) NOT NULL,
    image_path VARCHAR(1000),
    description TEXT,
    page_number INT,
    bbox JSON,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tables_data (
    id VARCHAR(36) PRIMARY KEY,
    chunk_id VARCHAR(36),
    document_id VARCHAR(36) NOT NULL,
    content TEXT NOT NULL,
    markdown_content TEXT,
    page_number INT,
    position INT,
    row_count INT,
    col_count INT,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
"""


def init():
    client = MySQLClient()
    for statement in TABLES_SQL.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            client.execute(stmt)
    print("MySQL tables created successfully.")


if __name__ == "__main__":
    init()
