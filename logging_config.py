import logging
import sys
import locale

from dotenv import load_dotenv

load_dotenv()

from browser_use.config import CONFIG
from browser_use.timing import now_utc_iso, uptime_seconds, process_start_utc_iso


def addLoggingLevel(levelName, levelNum, methodName=None):
	"""
	Comprehensively adds a new logging level to the `logging` module and the
	currently configured logging class.

	`levelName` becomes an attribute of the `logging` module with the value
	`levelNum`. `methodName` becomes a convenience method for both `logging`
	itself and the class returned by `logging.getLoggerClass()` (usually just
	`logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
	used.

	To avoid accidental clobberings of existing attributes, this method will
	raise an `AttributeError` if the level name is already an attribute of the
	`logging` module or if the method name is already present

	Example
	-------
	>>> addLoggingLevel('TRACE', logging.DEBUG - 5)
	>>> logging.getLogger(__name__).setLevel('TRACE')
	>>> logging.getLogger(__name__).trace('that worked')
	>>> logging.trace('so did this')
	>>> logging.TRACE
	5

	"""
	if not methodName:
		methodName = levelName.lower()

	if hasattr(logging, levelName):
		raise AttributeError(f'{levelName} already defined in logging module')
	if hasattr(logging, methodName):
		raise AttributeError(f'{methodName} already defined in logging module')
	if hasattr(logging.getLoggerClass(), methodName):
		raise AttributeError(f'{methodName} already defined in logger class')

	# This method was inspired by the answers to Stack Overflow post
	# http://stackoverflow.com/q/2183233/2988730, especially
	# http://stackoverflow.com/a/13638084/2988730
	def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(levelNum):
			self._log(levelNum, message, args, **kwargs)

	def logToRoot(message, *args, **kwargs):
		logging.log(levelNum, message, *args, **kwargs)

	logging.addLevelName(levelNum, levelName)
	setattr(logging, levelName, levelNum)
	setattr(logging.getLoggerClass(), methodName, logForLevel)
	setattr(logging, methodName, logToRoot)


class SafeStreamHandler(logging.StreamHandler):
	"""A logging handler that gracefully handles consoles that can't encode emojis.

	It retries writes with 'replace' on UnicodeEncodeError to avoid crashing tests on Windows cp1252.
	"""

	def emit(self, record):  # type: ignore[override]
		"""Emit a record, sanitizing any non-encodable characters proactively.

		Avoids calling the base StreamHandler.emit to prevent UnicodeEncodeError
		from escaping on Windows cp1252 consoles. Always writes a sanitized string.
		"""
		try:
			msg = self.format(record)
			stream = self.stream
			try:
				# Try a normal write first
				stream.write(msg + self.terminator)
			except UnicodeEncodeError:
				# Fallback: replace un-encodable characters
				enc = getattr(stream, "encoding", None) or locale.getpreferredencoding(False) or "utf-8"
				sanitized = msg.encode(enc, errors='replace').decode(enc, errors='replace')
				stream.write(sanitized + self.terminator)
			self.flush()
		except Exception:
			# Never let logging crash the app/tests
			try:
				stream = getattr(self, "stream", None)
				if stream is not None:
					stream.write("[log output suppressed due to logging error]\n")
					self.flush()
			except Exception:
				pass


def setup_logging(stream=None, log_level=None, force_setup=False):
	"""Setup logging configuration for browser-use.

	Args:
		stream: Output stream for logs (default: sys.stdout). Can be sys.stderr for MCP mode.
		log_level: Override log level (default: uses CONFIG.BROWSER_USE_LOGGING_LEVEL)
		force_setup: Force reconfiguration even if handlers already exist
	"""
	# Try to add RESULT level, but ignore if it already exists
	try:
		addLoggingLevel('RESULT', 35)  # This allows ERROR, FATAL and CRITICAL
	except AttributeError:
		pass  # Level already exists, which is fine

	log_type = log_level or CONFIG.BROWSER_USE_LOGGING_LEVEL

	# Check if handlers are already set up
	if logging.getLogger().hasHandlers() and not force_setup:
		return logging.getLogger('browser_use')

	# Clear existing handlers
	root = logging.getLogger()
	root.handlers = []

	class BrowserUseFormatter(logging.Formatter):
		def format(self, record):
			# Inject time context
			try:
				record.utc = now_utc_iso()  # e.g., 2025-08-25T12:34:56.789Z
				record.uptime = f"{uptime_seconds():.3f}s"
			except Exception:
				record.utc = ""
				record.uptime = ""
			return super().format(record)

	# Setup single handler for all loggers
	console = SafeStreamHandler(stream or sys.stdout)

	# adittional setLevel here to filter logs
	if log_type == 'result':
		console.setLevel('RESULT')
		console.setFormatter(BrowserUseFormatter('%(message)s'))
	else:
		# Show UTC timestamp and process uptime for every message
		console.setFormatter(BrowserUseFormatter('%(levelname)-8s [%(name)s] %(utc)s (+%(uptime)s) %(message)s'))

	# Configure root logger only
	root.addHandler(console)

	# switch cases for log_type
	if log_type == 'result':
		root.setLevel('RESULT')  # string usage to avoid syntax error
	elif log_type == 'debug':
		root.setLevel(logging.DEBUG)
	else:
		root.setLevel(logging.INFO)

	# Configure browser_use logger
	browser_use_logger = logging.getLogger('browser_use')
	browser_use_logger.propagate = False  # Don't propagate to root logger
	browser_use_logger.addHandler(console)
	browser_use_logger.setLevel(root.level)  # Set same level as root logger

	logger = logging.getLogger('browser_use')
	# logger.info('BrowserUse logging setup complete with level %s', log_type)
	try:
		logger.debug(f"Logging initialized at {now_utc_iso()} (process_start={process_start_utc_iso()})")
	except Exception:
		pass
	# Silence or adjust third-party loggers
	third_party_loggers = [
		'WDM',
		'httpx',
		'selenium',
		'playwright',
		'urllib3',
		'asyncio',
		'langsmith',
		'langsmith.client',
		'openai',
		'httpcore',
		'charset_normalizer',
		'anthropic._base_client',
		'PIL.PngImagePlugin',
		'trafilatura.htmlprocessing',
		'trafilatura',
		'groq',
		'portalocker',
		'portalocker.utils',
	]
	for logger_name in third_party_loggers:
		third_party = logging.getLogger(logger_name)
		third_party.setLevel(logging.ERROR)
		third_party.propagate = False

	return logger
