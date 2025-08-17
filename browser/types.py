# centralize imports for browser typing

import sys

try:
	from patchright._impl._errors import TargetClosedError as PatchrightTargetClosedError  # type: ignore
	from patchright.async_api import Browser as PatchrightBrowser  # type: ignore
	from patchright.async_api import BrowserContext as PatchrightBrowserContext  # type: ignore
	from patchright.async_api import ElementHandle as PatchrightElementHandle  # type: ignore
	from patchright.async_api import FrameLocator as PatchrightFrameLocator  # type: ignore
	from patchright.async_api import Page as PatchrightPage  # type: ignore
	from patchright.async_api import Playwright as Patchright  # type: ignore
	from patchright.async_api import async_playwright as _async_patchright  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
	PatchrightTargetClosedError = type("PatchrightTargetClosedError", (Exception,), {})  # type: ignore
	PatchrightBrowser = object  # type: ignore
	PatchrightBrowserContext = object  # type: ignore
	PatchrightElementHandle = object  # type: ignore
	PatchrightFrameLocator = object  # type: ignore
	PatchrightPage = object  # type: ignore
	class Patchright:  # type: ignore
		pass
	def _async_patchright():  # type: ignore
		raise RuntimeError("patchright is not available")

try:
	from playwright._impl._errors import TargetClosedError as PlaywrightTargetClosedError  # type: ignore
	from playwright.async_api import Browser as PlaywrightBrowser  # type: ignore
	from playwright.async_api import BrowserContext as PlaywrightBrowserContext  # type: ignore
	from playwright.async_api import ElementHandle as PlaywrightElementHandle  # type: ignore
	from playwright.async_api import FrameLocator as PlaywrightFrameLocator  # type: ignore
	from playwright.async_api import Page as PlaywrightPage  # type: ignore
	from playwright.async_api import Playwright as Playwright  # type: ignore
	from playwright.async_api import async_playwright as _async_playwright  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
	PlaywrightTargetClosedError = type("PlaywrightTargetClosedError", (Exception,), {})  # type: ignore
	PlaywrightBrowser = object  # type: ignore
	PlaywrightBrowserContext = object  # type: ignore
	PlaywrightElementHandle = object  # type: ignore
	PlaywrightFrameLocator = object  # type: ignore
	PlaywrightPage = object  # type: ignore
	class Playwright:  # type: ignore
		pass
	def _async_playwright():  # type: ignore
		raise RuntimeError("playwright is not available")

# Define types to be Union[Patchright, Playwright]
Browser = PatchrightBrowser | PlaywrightBrowser  # type: ignore
BrowserContext = PatchrightBrowserContext | PlaywrightBrowserContext  # type: ignore
Page = PatchrightPage | PlaywrightPage  # type: ignore
ElementHandle = PatchrightElementHandle | PlaywrightElementHandle  # type: ignore
FrameLocator = PatchrightFrameLocator | PlaywrightFrameLocator  # type: ignore
Playwright = Playwright  # type: ignore
Patchright = Patchright  # type: ignore
PlaywrightOrPatchright = Patchright | Playwright  # type: ignore
TargetClosedError = PatchrightTargetClosedError | PlaywrightTargetClosedError  # type: ignore

async_patchright = _async_patchright
async_playwright = _async_playwright

try:
	from playwright._impl._api_structures import (  # type: ignore
		ClientCertificate,
		Geolocation,
		HttpCredentials,
		ProxySettings,
		StorageState,
		ViewportSize,
	)
except Exception:  # pragma: no cover - optional dependency fallback
	# Provide minimal structural stand-ins for pydantic typing using TypedDicts
	try:
		from typing_extensions import TypedDict  # type: ignore
	except Exception:
		from typing import TypedDict  # type: ignore

	class ClientCertificate(TypedDict, total=False):  # type: ignore
		pass

	class Geolocation(TypedDict, total=False):  # type: ignore
		latitude: float
		longitude: float
		accuracy: float

	class HttpCredentials(TypedDict, total=False):  # type: ignore
		username: str
		password: str

	class ProxySettings(TypedDict, total=False):  # type: ignore
		server: str
		username: str
		password: str

	class StorageState(TypedDict, total=False):  # type: ignore
		cookies: list[dict]
		origins: list[dict]

	class ViewportSize(TypedDict, total=False):  # type: ignore
		width: int
		height: int

# fix pydantic error on python 3.11
# PydanticUserError: Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.
# For further information visit https://errors.pydantic.dev/2.10/u/typed-dict-version
if sys.version_info < (3, 12):
	from typing_extensions import TypedDict

	# convert new-style typing.TypedDict used by playwright to old-style typing_extensions.TypedDict used by pydantic
	ClientCertificate = TypedDict('ClientCertificate', ClientCertificate.__annotations__, total=ClientCertificate.__total__)
	Geolocation = TypedDict('Geolocation', Geolocation.__annotations__, total=Geolocation.__total__)
	ProxySettings = TypedDict('ProxySettings', ProxySettings.__annotations__, total=ProxySettings.__total__)
	ViewportSize = TypedDict('ViewportSize', ViewportSize.__annotations__, total=ViewportSize.__total__)
	HttpCredentials = TypedDict('HttpCredentials', HttpCredentials.__annotations__, total=HttpCredentials.__total__)
	StorageState = TypedDict('StorageState', StorageState.__annotations__, total=StorageState.__total__)
