from django.db import models

# Create your models here.
class Core(models.Model):
    """core definition"""

    name = models.CharField(max_length=30, null=False)

    class Meta:
        ordering = ("-id",)


class AbstractTimeStampedModel(models.Model):
    """Abstract Model with Time Stamps"""

    created = models.DateTimeField(auto_now_add=True, editable=False)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
