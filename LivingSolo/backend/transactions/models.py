from django.db import models
from core.models import AbstractTimeStampedModel


class TransactionType(AbstractTimeStampedModel):
    """Transaction Type definition"""

    name = models.CharField(max_length=30, null=False)
    color = models.CharField(max_length=7, null=False)


# Create your models here.
class Transaction(AbstractTimeStampedModel):
    """Transaction definition"""

    date = models.DateTimeField(blank=True)
    type = models.ManyToManyField(TransactionType, related_name="transaction")
    amount = models.IntegerField()
    memo = models.TextField()

    period = models.IntegerField(default=0) # 0 means Not Periodic Transaction.


    class Meta:
        ordering = ("-date",)
