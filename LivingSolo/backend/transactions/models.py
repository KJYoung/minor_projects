from django.db import models

# Create your models here.
class Transaction(models.Model):
    """Transaction definition"""

    date = models.DateTimeField()
    type = models.CharField(max_length=30, null=False)
    amount = models.IntegerField()
    memo = models.TextField()

    class Meta:
        ordering = ("-date",)
