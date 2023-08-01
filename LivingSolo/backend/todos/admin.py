"""
    TodoCategory, Todo, 할일에 관한 어드민 패널입니다.
"""
from django.contrib import admin
from todos.models import TodoCategory, Todo


@admin.register(TodoCategory)
class TodoCategoryAdmin(admin.ModelAdmin):
    """TodoCategory admin definition"""

    list_display = ("name", "color", "pk")


@admin.register(Todo)
class TodoAdmin(admin.ModelAdmin):
    """Todo admin definition"""

    list_display = ("name", "done", "category", "pk", "priority", "deadline")
