from django.db import models

class Data_tg(models.Model):
    chanel = models.CharField(max_length=100)
    text = models.TextField(max_length=500)
    date_time = models.DateTimeField(auto_now=True)