from django.db import models
from core.models import AbstractTimeStampedModel


class TransactionType(AbstractTimeStampedModel):
    """Transaction Type definition"""

    name = models.CharField(max_length=30, null=False)


# Create your models here.
class Transaction(AbstractTimeStampedModel):
    """Transaction definition"""

    date = models.DateTimeField()
    type = models.ForeignKey(
        TransactionType, on_delete=models.SET_NULL, related_name="transaction", null=True
    )
    amount = models.IntegerField()
    memo = models.TextField()

    class Meta:
        ordering = ("-date",)
