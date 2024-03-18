from django.db import models


# Create your models here.
class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name


class Word(models.Model):
    word = models.CharField(max_length=100)
    file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE)
    left_context = models.TextField()
    right_context = models.TextField()

    def __str__(self):
        return self.word
