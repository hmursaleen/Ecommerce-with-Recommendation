# Generated by Django 3.2.20 on 2023-08-31 09:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_auto_20230831_1520'),
    ]

    operations = [
        migrations.AlterField(
            model_name='product',
            name='status',
            field=models.CharField(choices=[('draft', 'Draft'), ('active', 'Active'), ('waitingapproval', 'Waiting approval'), ('deleted', 'Deleted')], default='active', max_length=50),
        ),
    ]