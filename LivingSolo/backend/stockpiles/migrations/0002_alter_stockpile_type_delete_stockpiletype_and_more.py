# Generated by Django 4.1.3 on 2023-07-20 17:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("tags", "0001_initial"),
        ("stockpiles", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="stockpile",
            name="type",
            field=models.ManyToManyField(related_name="stockpile", to="tags.tag"),
        ),
        migrations.DeleteModel(
            name="StockpileType",
        ),
        migrations.DeleteModel(
            name="StockpileTypeClass",
        ),
    ]
