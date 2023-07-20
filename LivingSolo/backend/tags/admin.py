"""
    TagClass, Tag, 태그에 관한 어드민 패널입니다.
"""
from django.contrib import admin
from tags.models import TagClass, Tag


@admin.register(TagClass)
class TagClassAdmin(admin.ModelAdmin):
    """TagClass admin definition"""

    list_display = ("name", "color", "pk")


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    """Tag admin definition"""

    list_display = ("name", "color", "tag_class", "pk")
