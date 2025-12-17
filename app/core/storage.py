from __future__ import annotations

import os
import shutil
from typing import Protocol, Optional, TYPE_CHECKING

from app.core.logger import get_logger, log_exception

if TYPE_CHECKING:
    # Only imported for type checking; avoids hard dependency at runtime
    from google.auth.credentials import Credentials  # type: ignore

logger = get_logger(__name__)


class OutputSink(Protocol):
    def write(self, local_tmp_path: str, dest_name: str) -> str: ...


class LocalSink:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        # Ensure destination directory exists
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            # Log and re-raise to surface configuration errors early
            log_exception("LOCAL_SINK_INIT", e)
            raise

    def write(self, local_tmp_path: str, dest_name: str) -> str:
        # Preserve original extension if dest_name has none
        source_ext = os.path.splitext(local_tmp_path)[1]
        dest_ext = os.path.splitext(dest_name)[1]
        
        # CRITICAL FIX: Always ensure extension is preserved for document types
        # Log for debugging extension issues
        logger.debug(f"LocalSink.write: source={local_tmp_path}, dest_name={dest_name}, source_ext={source_ext}, dest_ext={dest_ext}")
        
        if not dest_ext and source_ext:
            dest_name = dest_name + source_ext
            logger.debug(f"LocalSink.write: Added source extension to dest_name: {dest_name}")
        elif dest_ext and source_ext and dest_ext.lower() != source_ext.lower():
            # Extension mismatch - prefer source extension for document types
            if source_ext.lower() in ['.docx', '.doc', '.pdf', '.md', '.txt']:
                logger.warning(f"LocalSink.write: Extension mismatch! source={source_ext}, dest={dest_ext}. Using source extension.")
                dest_name = os.path.splitext(dest_name)[0] + source_ext
        
        dest = os.path.join(self.output_dir, dest_name)
        logger.debug(f"LocalSink.write: Final destination path: {dest}")

        # Validate source file exists
        if not os.path.exists(local_tmp_path):
            raise FileNotFoundError(f"Source file does not exist: {local_tmp_path}")

        # Generate unique filename if destination exists
        if os.path.exists(dest):
            base_name, ext = os.path.splitext(dest_name)
            counter = 1
            while os.path.exists(dest):
                # Create variation with similarity indicator
                variation_name = f"{base_name}_sim{counter}{ext}"
                dest = os.path.join(self.output_dir, variation_name)
                counter += 1
                # Prevent infinite loop
                if counter > 1000:
                    logger.warning(f"Too many similar files for {dest_name}, using timestamp")
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    variation_name = f"{base_name}_sim_{timestamp}{ext}"
                    dest = os.path.join(self.output_dir, variation_name)
                    break
            logger.debug(f"Generated unique filename: {os.path.basename(dest)}")

        # Perform atomic move into destination directory
        # Use copy + delete instead of os.replace to handle cross-device links
        # (e.g., when moving from /tmp to /var/task on Vercel)
        try:
            # Try atomic move first (fastest, works on same filesystem)
            try:
                os.replace(local_tmp_path, dest)
                return dest
            except OSError as e:
                # If cross-device link error (errno 18), fall back to copy + delete
                if e.errno == 18:  # Invalid cross-device link
                    logger.debug(f"Cross-device link detected, using copy+delete: {local_tmp_path} -> {dest}")
                    shutil.copy2(local_tmp_path, dest)
                    os.remove(local_tmp_path)
                    return dest
                else:
                    # Re-raise other OSErrors
                    raise
        except Exception as e:
            log_exception("LOCAL_SINK_WRITE", e)
            raise


class DriveSink:
    def __init__(self, folder_id: str, creds: "Credentials") -> None:
        self.folder_id = folder_id
        self.creds = creds

    def write(self, local_tmp_path: str, dest_name: str) -> str:
        from app.utils.utils import create_google_doc

        if not os.path.exists(local_tmp_path):
            raise FileNotFoundError(f"Source file does not exist: {local_tmp_path}")

        # Attempt upload; ensure local temp cleanup on success
        try:
            doc_id = create_google_doc(local_tmp_path, dest_name, self.folder_id, self.creds)
        except Exception as e:
            log_exception("DRIVE_SINK_UPLOAD", e)
            raise RuntimeError(f"Failed to upload '{dest_name}' to Google Drive: {e}") from e
        else:
            # Best-effort cleanup of local temp file after successful upload
            try:
                os.remove(local_tmp_path)
            except Exception:
                # Non-fatal; keep going if cleanup fails
                pass
            return doc_id


