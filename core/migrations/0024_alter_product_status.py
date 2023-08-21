# Generated by Django 4.2.4 on 2023-08-19 21:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0023_alter_product_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='product',
            name='status',
            field=models.CharField(choices=[('deleted', 'Deleted'), ('active', 'Active'), ('draft', 'Draft'), ('waitingapproval', 'Waiting approval')], default='active', max_length=50),
        ),
    ]