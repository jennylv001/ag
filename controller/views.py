from typing import (
    Any, Dict, Generic, List, Optional, Type, TypeVar, Union, TYPE_CHECKING
)

from pydantic import BaseModel, ConfigDict, Field


# Action Input Models
class SearchGoogleAction(BaseModel):
	query: str


class GoToUrlAction(BaseModel):
	url: str
	new_tab: bool  # True to open in new tab, False to navigate in current tab


class ClickElementAction(BaseModel):
	index: int


class InputTextAction(BaseModel):
	index: int
	text: str


# Completion Actions
T = TypeVar('T', bound=BaseModel)


class DoneAction(BaseModel):
    text: str = Field(..., description="The final summary or answer for the user.")
    success: bool = Field(..., description="Whether the task was completed successfully.")
    files_to_display: Optional[List[str]] = Field(None, description="A list of files to display with the final result.")


class StructuredOutputAction(BaseModel, Generic[T]):
    success: bool = Field(True, description="Whether the task was completed successfully.")
    data: T = Field(..., description="The structured data object as the final result.")


class SwitchTabAction(BaseModel):
	page_id: int


class CloseTabAction(BaseModel):
	page_id: int


class ScrollAction(BaseModel):
	down: bool  # True to scroll down, False to scroll up
	num_pages: float  # Number of pages to scroll (0.5 = half page, 1.0 = one page, etc.)
	index: int | None = None  # Optional element index to find scroll container for


class SendKeysAction(BaseModel):
	keys: str


class UploadFileAction(BaseModel):
	index: int
	path: str


class ExtractPageContentAction(BaseModel):
	value: str


class NoParamsAction(BaseModel):
	"""
	Accepts absolutely anything in the incoming data
	and discards it, so the final parsed model is empty.
	"""

	model_config = ConfigDict(extra='ignore')
	# No fields defined - all inputs are ignored automatically

class GetDropdownOptionsAction(BaseModel):
    index: int = Field(..., description="The index of the dropdown element.")


class SelectDropdownOptionAction(BaseModel):
    index: int = Field(..., description="The index of the dropdown element.")
    text: str = Field(..., description="The visible text of the option to select.")


# Content & Utility Actions
class WaitAction(BaseModel):
    seconds: int = Field(3, description="The number of seconds to wait.")


class ScrollToTextAction(BaseModel):
    text: str = Field(..., description="The text to scroll to on the page.")

class Position(BaseModel):
	x: int
	y: int


class DragDropAction(BaseModel):
	# Element-based approach
	element_source: str | None = Field(None, description='CSS selector or XPath of the element to drag from')
	element_target: str | None = Field(None, description='CSS selector or XPath of the element to drop onto')
	element_source_offset: Position | None = Field(
		None, description='Precise position within the source element to start drag (in pixels from top-left corner)'
	)
	element_target_offset: Position | None = Field(
		None, description='Precise position within the target element to drop (in pixels from top-left corner)'
	)

	# Coordinate-based approach (used if selectors not provided)
	coord_source_x: int | None = Field(None, description='Absolute X coordinate on page to start drag from (in pixels)')
	coord_source_y: int | None = Field(None, description='Absolute Y coordinate on page to start drag from (in pixels)')
	coord_target_x: int | None = Field(None, description='Absolute X coordinate on page to drop at (in pixels)')
	coord_target_y: int | None = Field(None, description='Absolute Y coordinate on page to drop at (in pixels)')

	# Common options
	steps: int | None = Field(10, description='Number of intermediate points for smoother movement (5-20 recommended)')
	delay_ms: int | None = Field(5, description='Delay in milliseconds between steps (0 for fastest, 10-20 for more natural)')

class ExtractStructuredDataAction(BaseModel):
    query: str = Field(..., description="The natural language query for the information to extract.")
    extract_links: bool = Field(False, description="Set to True to include hyperlinks in the extracted text.")


# Memory primitives
class ExtractToMemoryAction(BaseModel):
	key: str = Field(..., description="Memory key to store the extracted value under.")
	source: str = Field(..., description="One of: 'url', 'title', 'selection', or 'css:<selector>' to extract from a specific element.")


class FillFromMemoryAction(BaseModel):
	field_selector: str = Field(..., description="Target input field to fill. Form 'css:<selector>' or 'xpath:<expr>'.")
	key: str = Field(..., description="Memory key to read.")


# File System Actions
class WriteFileAction(BaseModel):
    file_name: str = Field(..., description="The name of the file to write to.")
    content: str = Field(..., description="The content to write to the file.")


class AppendFileAction(BaseModel):
    file_name: str = Field(..., description="The name of the file to append to.")
    content: str = Field(..., description="The content to append to the file.")


class ReadFileAction(BaseModel):
    file_name: str = Field(..., description="The name of the file to read.")


# Google Sheets Actions
class ReadSheetContentsAction(NoParamsAction):
    pass


class ReadCellContentsAction(BaseModel):
    cell_or_range: str = Field(..., description='The cell (e.g., "A1") or range (e.g., "A1:B5") to read.')


class UpdateCellContentsAction(BaseModel):
    cell_or_range: str = Field(..., description="The cell or range to update.")
    new_contents_tsv: str = Field(..., description="The new content in Tab-Separated Value (TSV) format.")


class ClearCellContentsAction(BaseModel):
    cell_or_range: str = Field(..., description="The cell or range to clear.")


class SelectCellOrRangeAction(BaseModel):
    cell_or_range: str = Field(..., description="The cell or range to select.")


class FallbackInputIntoSingleSelectedCellAction(BaseModel):
    text: str = Field(..., description="The text to type into the currently selected cell.")


