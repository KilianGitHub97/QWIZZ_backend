import os

from django.contrib.auth.management.commands import createsuperuser
from django.core.management import CommandError


class Command(createsuperuser.Command):
    help = "Crate a superuser, and allow password to be provided"

    def add_arguments(self, parser):
        super(Command, self).add_arguments(parser)
        parser.add_argument(
            "--password",
            dest="password",
            default=None,
            help="Specifies the password for the superuser.",
        )
        parser.add_argument(
            "--preserve",
            dest="preserve",
            default=False,
            action="store_true",
            help="Exit normally if the user already exists.",
        )

    def handle(self, *args, **options):
        password = options.get("password")
        username = options.get("username")
        database = options.get("database")
        email = os.environ.get("DJANGO_SUPERUSER_EMAIL")

        if password and not username:
            raise CommandError(
                "--username is required if specifying --password"
            )

        if email and options.get("preserve"):
            exists = (
                self.UserModel._default_manager.db_manager(database)
                .filter(email=email)
                .exists()
            )
            if exists:
                self.stdout.write(
                    "User exists, exiting normally due to --preserve"
                )
                return

        super(Command, self).handle(*args, **options)

        if password:
            user = self.UserModel._default_manager.db_manager(database).get(
                username=username
            )
            user.set_password(password)
            user.save()
