from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
import hashlib
import pyodbc

class PDFInfo:
    def __init__(self, file_name: str, customerNumber: str | None = None, addressNumber: str | None = None):
        self.file_name = file_name
        self.customerNumber = customerNumber
        self.addressNumber = addressNumber

class ProcessIndexSqlServer:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    @contextmanager
    def _conn(self):
        con = pyodbc.connect(self.connection_string, autocommit=False)
        try:
            yield con
            con.commit()
        except:
            con.rollback()
            raise
        finally:
            con.close()

    @staticmethod
    def _now_dt() -> datetime:
        return datetime.now()

    def upsert_discovered(self, full_path: str, customerNumber: str, addressNumber: str):
        """
        Stores only FileName (not full path). CustomerNumber + FileName identifies the file.
        Drops fingerprinting entirely; only stores size and metadata you actually have.
        """
        file_name = Path(full_path).name
        size = Path(full_path).stat().st_size
        now = self._now_dt()

        with self._conn() as con:
            cur = con.cursor()

            # nchar in SQL Server pads; compare using RTRIM to avoid whitespace issues
            existing = cur.execute("""
                SELECT TOP 1 RTRIM([status]) AS status
                FROM dbo.Files_Production
                WHERE RTRIM(CustomerNumber) = ? AND FileName = ?
                ORDER BY ID DESC
            """, (customerNumber, file_name)).fetchone()

            if existing:
                # Refresh metadata; keep status as-is
                cur.execute("""
                    UPDATE dbo.Files_Production
                    SET [size] = ?, AddressNumber = ?
                    WHERE RTRIM(CustomerNumber) = ? AND FileName = ?
                """, (size, addressNumber, customerNumber, file_name))
                return existing[0]

            cur.execute("""
                INSERT INTO dbo.Files_Production
                    (FileName, [size], [status], LastError, CustomerNumber, AddressNumber, DiscoveredAt, ProcessedAt)
                VALUES
                    (?, ?, ?, NULL, ?, ?, ?, NULL)
            """, (file_name, size, "queued", customerNumber, addressNumber, now))

            return "queued"

    def is_already_processed(self, full_path: str, customerNumber: str) -> bool:
        file_name = Path(full_path).name
        with self._conn() as con:
            row = con.cursor().execute("""
                SELECT TOP 1 RTRIM([status]) AS status
                FROM dbo.Files_Production
                WHERE RTRIM(CustomerNumber) = ? AND FileName = ?
                ORDER BY ID DESC
            """, (customerNumber, file_name)).fetchone()
            return bool(row and row[0] == "success")

    def mark_processing(self, full_path: str, customerNumber: str):
        file_name = Path(full_path).name
        with self._conn() as con:
            con.cursor().execute("""
                UPDATE dbo.Files_Production
                SET [status] = ?, LastError = NULL
                WHERE RTRIM(CustomerNumber) = ? AND FileName = ?
            """, ("processing", customerNumber, file_name))

    def mark_result(self, full_path: str, customerNumber: str, success: bool, error: str | None):
        file_name = Path(full_path).name
        status = "success" if success else "error"
        now = self._now_dt()

        with self._conn() as con:
            con.cursor().execute("""
                UPDATE dbo.Files_Production
                SET [status] = ?, LastError = ?, ProcessedAt = ?
                WHERE RTRIM(CustomerNumber) = ? AND FileName = ?
            """, (status, error, now, customerNumber, file_name))

    def get_unprocessed_candidates(self) -> list[PDFInfo]:
        with self._conn() as con:
            rows = con.cursor().execute("""
                SELECT FileName, CustomerNumber, AddressNumber
                FROM dbo.Files_Production
                WHERE RTRIM([status]) = ?
            """, ("queued",)).fetchall()

        return [
            PDFInfo(
                r[0],
                r[1].strip() if isinstance(r[1], str) else r[1],
                r[2].strip() if isinstance(r[2], str) else r[2],
            )
            for r in rows
        ]