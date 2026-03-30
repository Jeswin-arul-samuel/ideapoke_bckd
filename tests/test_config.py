from app.config import settings


def test_settings_loads_database_url():
    assert settings.DATABASE_URL is not None
    assert "postgresql" in settings.DATABASE_URL


def test_settings_loads_openai_key():
    assert hasattr(settings, "OPENAI_API_KEY")


def test_settings_loads_serp_key():
    assert hasattr(settings, "SERP_API_KEY")
