from django import forms


class QuestionForm(forms.Form):
    question = forms.CharField(
        label="",
        max_length=600,
        widget=forms.Textarea(attrs={"rows": 8, "cols": 70})
    )
    