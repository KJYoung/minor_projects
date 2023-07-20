"""
    Transaction, Trxn, 거래에 관한 모델입니다.
"""
from django.db import models
from core.models import AbstractTimeStampedModel
from tags.models import Tag
from stockpiles.models import StockpileTransaction

PAY_TYPE_CASH = "현금"
PAY_TYPE_CHECK = "체크카드"
PAY_TYPE_CREDIT = "신용카드"
PAY_TYPE_PAY = "페이포인트"

PAY_TYPE_CHOICES = (
    (PAY_TYPE_CASH, "현금"),
    (PAY_TYPE_CHECK, "체크카드"),
    (PAY_TYPE_CREDIT, "신용카드"),
    (PAY_TYPE_PAY, "페이포인트"),
)


class PayMethod(AbstractTimeStampedModel):
    type = models.CharField(max_length=20, choices=PAY_TYPE_CHOICES, default=PAY_TYPE_CASH)
    name = models.CharField(max_length=30, null=False)


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

    method = models.ForeignKey(
        PayMethod, related_name='transaction', on_delete=models.SET_NULL, null=True
    )

    class Meta:
        ordering = ("-date",)
