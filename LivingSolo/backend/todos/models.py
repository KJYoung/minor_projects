"""
    Todo, 할일에 관한 모델입니다.
"""
from django.db import models
from core.models import AbstractTimeStampedModel
from tags.models import Tag

# Create your models here.
class TodoCategory(AbstractTimeStampedModel):
    """Todo Category definition"""

    name = models.CharField(max_length=16, null=False)
    color = models.CharField(max_length=7, null=False)
    tag = models.ManyToManyField(Tag, related_name="todoCategory")

    def __str__(self):
        """To string method"""
        return str(self.name)

    class Meta:
        verbose_name_plural = "Todo Categories"


class Todo(AbstractTimeStampedModel):
    """Todo definition"""

    name = models.CharField(max_length=24, null=False)
    done = models.BooleanField(default=False)
    category = models.ForeignKey(
        TodoCategory, related_name='todo', on_delete=models.SET_NULL, null=True
    )
    tag = models.ManyToManyField(Tag, related_name="todo")

    priority = models.IntegerField()
    deadline = models.DateTimeField(blank=True)
    is_hard_deadline = models.BooleanField(default=False)
    period = models.IntegerField()

    def __str__(self):
        """To string method"""
        return str(self.name)
