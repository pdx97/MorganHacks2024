from django.db import models

class User(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    job_titles = models.CharField(max_length=255)
    job_description = models.CharField(max_length=255)
    skills = models.TextField(help_text="List of skills separated by commas")
    experience = models.IntegerField(help_text="Experience in years")
    education = models.TextField(help_text="Highest level of education or degrees obtained")
