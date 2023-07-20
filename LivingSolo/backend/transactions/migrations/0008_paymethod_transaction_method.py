# Generated by Django 4.1.3 on 2023-07-21 00:35

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("transactions", "0007_rename_type_transaction_tag"),
    ]

    operations = [
        migrations.CreateModel(
            name="PayMethod",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "created",
                    models.DateTimeField(
                        default=django.utils.timezone.now, editable=False
                    ),
                ),
                ("updated", models.DateTimeField(auto_now=True)),
                (
                    "type",
                    models.CharField(
                        choices=[
                            ("현금", "현금"),
                            ("체크카드", "체크카드"),
                            ("신용카드", "신용카드"),
                            ("페이포인트", "페이포인트"),
                        ],
                        default="현금",
                        max_length=20,
                    ),
                ),
                ("name", models.CharField(max_length=30)),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.AddField(
            model_name="transaction",
            name="method",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="transaction",
                to="transactions.paymethod",
            ),
        ),
    ]
