from django.contrib import admin
from core.models import Core


@admin.register(Core)
class CoreAdmin(admin.ModelAdmin):
    """Core admin definition"""

    list_display = ("pk", "name")
