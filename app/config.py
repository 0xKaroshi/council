"""council — runtime settings.

Loads from a `.env` file (via pydantic-settings) and environment
variables. All settings have sensible defaults so the local CLI
works on a fresh checkout once the user supplies an OpenAI key.
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Local CLI mode (default) -------------------------------------
    # Where per-mentor SQLite DBs live. Overridable via COUNCIL_DATA_DIR.
    data_dir: Path = Field(default=Path("./data"), validation_alias="COUNCIL_DATA_DIR")

    # User-context markdown files (brand profile, business situation,
    # constraints, etc). Overridable via COUNCIL_USER_CONTEXT_DIR.
    user_context_dir: Path = Field(
        default=Path("./user_context"),
        validation_alias="COUNCIL_USER_CONTEXT_DIR",
    )

    # YAML config that defines the mentor lineup. Overridable via
    # COUNCIL_CONFIG. (Also read directly by app/ingest/mentors.py.)
    config_path: Path = Field(
        default=Path("./config/mentors.yaml"),
        validation_alias="COUNCIL_CONFIG",
    )

    # ---- API credentials ----------------------------------------------
    # Required for embedding (text-embedding-3-small, $0.02 / 1M tokens).
    openai_api_key: str = Field(default="")

    # Optional. Required only for the twitter source. Sign up at
    # twitterapi.io; ~$0.15 per 1,000 tweets at current pricing.
    twitter_api_key: str = Field(
        default="",
        validation_alias="TWITTER_API_KEY",
    )

    # ---- MCP server mode (advanced, optional) -------------------------
    # The CLI never needs these; they apply only when running
    # `uvicorn app.main:app` to expose an MCP endpoint.

    # "none" = open server (no auth — only safe behind your own
    #          reverse proxy or on localhost).
    # "oauth" = OAuth 2.1 / DCR / PKCE gate (NOT shipped in council v1;
    #           future work — see CONTRIBUTING.md).
    mcp_auth_mode: str = Field(default="none")
    mcp_port: int = Field(default=8440)

    # Public base URL the MCP server advertises (in OAuth discovery
    # docs and the FastAPI host header check).
    public_url: str = Field(default="http://localhost:8440")

    # FastMCP's DNS-rebinding check: must include the public host.
    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"]
    )

    # ---- Internal --------------------------------------------------
    env: str = Field(default="development")

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [h.strip() for h in v.split(",") if h.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


settings = Settings()
