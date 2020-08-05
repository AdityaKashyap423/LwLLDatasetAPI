import firebase_admin
from firebase_admin import credentials, firestore
import os


class FB_Auth_Public(object):
    def __init__(self) -> None:
        # Use a service account
        cred = credentials.Certificate(
            f'{os.path.dirname(__file__)}/service_accounts/creds.json')
        _ = firebase_admin.initialize_app(cred)

        # First initialization can't have `name` argument, should really file bug on python SDK of firebase_admin repo
        self.db_firestore = firestore.client()
        self.db_realtime = ""
        pass

class FB_Auth_Private(object):
    def __init__(self) -> None:
        # Use a service account
        cred = credentials.Certificate(
            f'{os.path.dirname(__file__)}/service_accounts/creds2.json')
        app2 = firebase_admin.initialize_app(cred, name='fb_private')

        self.db_firestore = firestore.client(app2)
        self.db_realtime = ""
        pass


firebase = FB_Auth_Public()
fb_store_public = firebase.db_firestore
fb_realtime_public = firebase.db_realtime

firebase2 = FB_Auth_Private()
fb_store_private = firebase2.db_firestore
fb_realtime_private = firebase2.db_realtime
