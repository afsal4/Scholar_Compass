from django.db import models

# db.execute("""
# CREATE TABLE IF NOT EXISTS chat_history (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     session_id TEXT,
#     role TEXT,
#     message TEXT,
#     timestamp TEXT
# );
# """)


# db.execute("""
# CREATE TABLE IF NOT EXISTS chatpdf_history (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     session_id TEXT,
#     role TEXT,
#     message TEXT,
#     timestamp TEXT
# );
# """)

# db.execute("""
# CREATE TABLE IF NOT EXISTS rag_history (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     session_id TEXT,
#     role TEXT,
#     message TEXT,
#     timestamp TEXT
# );
# """)


class ChatHistory(models.Model):
    session_id = models.CharField(max_length=100)
    role = models.CharField(max_length=10)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session_id} - {self.role} - {self.timestamp}"


class ChatPDFHistory(models.Model):
    session_id = models.CharField(max_length=100)
    role = models.CharField(max_length=10)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session_id} - {self.role} - {self.timestamp}"


class RagHistory(models.Model):
    session_id = models.CharField(max_length=100)
    role = models.CharField(max_length=10)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session_id} - {self.role} - {self.timestamp}"
