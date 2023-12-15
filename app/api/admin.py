import pprint

from django.contrib import admin
from django.contrib.sessions.models import Session

from .models import Chat, Document, DocumentQnA, Message, Note, Project


class SessionAdmin(admin.ModelAdmin):
    def _session_data(self, obj):
        return pprint.pformat(obj.get_decoded()).replace("\n", "<br>\n")

    _session_data.allow_tags = True  # type: ignore
    list_display = ["session_key", "_session_data", "expire_date"]
    readonly_fields = ["_session_data"]
    exclude = ["session_data"]
    date_hierarchy = "expire_date"


admin.site.register(Session, SessionAdmin)
admin.site.register(Document)
admin.site.register(Project)
admin.site.register(Chat)
admin.site.register(Message)
admin.site.register(Note)
admin.site.register(DocumentQnA)
