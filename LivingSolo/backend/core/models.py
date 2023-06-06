from django.db import models

# Create your models here.
class Core(models.Model):
    """core definition"""

    name = models.CharField(max_length=30, null=False)

    class Meta:
        ordering = ("-id",)