from django.db import models

class User(models.Model):
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=100)
    phone_number = models.IntegerField()
    job_title = models.CharField(max_length=255)
    targetJob = models.CharField(max_length=255)
    skills = models.TextField(help_text="List of skills separated by commas")
    experience = models.IntegerField(help_text="Experience in years")
    education = models.TextField(help_text="Highest level of education or degrees obtained")
