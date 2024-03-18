from django.contrib import admin
from .models import UploadedFile, Word


# Register your models here.
'''
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = ('file', 'uploaded_at')
    
    
    admin.site.register(UploadedFile, UploadedFileAdmin)
'''

'''
class WordAdmin(admin.ModelAdmin):
    list_display = ('id', 'word', 'file', 'context')
    list_filter = ('file',)
    search_fields = ('word', 'context')


    admin.site.register(Word, WordAdmin)
'''

admin.site.register(UploadedFile)
admin.site.register(Word)



