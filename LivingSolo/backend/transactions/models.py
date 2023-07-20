from django.db import models
from core.models import AbstractTimeStampedModel
from tags.models import Tag
from stockpiles.models import StockpileTransaction


class Transaction(AbstractTimeStampedModel):
    """Transaction definition"""

    date = models.DateTimeField(blank=True)
    tag = models.ManyToManyField(Tag, related_name="transaction")
    amount = models.IntegerField()
    memo = models.CharField(max_length=200, null=False)

    period = models.IntegerField(default=0)  # 0 means Not Periodic Transaction.

    stockpile_transaction = models.ManyToManyField(
        StockpileTransaction, related_name="mother_trxn"
    )  # Related Stockpile Change

    class Meta:
        ordering = ("-date",)
