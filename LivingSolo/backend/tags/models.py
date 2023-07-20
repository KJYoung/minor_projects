from django.db import models
from core.models import AbstractTimeStampedModel

# Create your models here.
class TagClass(AbstractTimeStampedModel):
    """Tag Class definition"""

    name = models.CharField(max_length=30, null=False)
    color = models.CharField(max_length=7, null=False)

    def __str__(self):
        """To string method"""
        return str(self.name)


class Tag(AbstractTimeStampedModel):
    """Tag definition"""

    name = models.CharField(max_length=30, null=False)
    color = models.CharField(max_length=7, null=False)
    tag_class = models.ForeignKey(TagClass, related_name='tag', on_delete=models.CASCADE, null=True)

    def __str__(self):
        """To string method"""
        return str(self.name)
