# Generated by Django 4.2.4 on 2023-08-16 18:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0008_userprofile'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='status',
            field=models.CharField(choices=[('active', 'Active'), ('waitingapproval', 'Waiting approval'), ('deleted', 'Deleted'), ('draft', 'Draft')], default='active', max_length=50),
        ),
    ]