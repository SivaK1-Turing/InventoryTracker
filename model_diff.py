# inventorytracker/utils/model_diff.py
from enum import Enum
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Set
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class DiffType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"

class FieldDiff:
    """Represents a difference in a single field between two models."""
    def __init__(self, field_name: str, old_value: Any, new_value: Any, diff_type: DiffType):
        self.field_name = field_name
        self.old_value = old_value
        self.new_value = new_value
        self.diff_type = diff_type
    
    def __str__(self) -> str:
        if self.diff_type == DiffType.UNCHANGED:
            return f"{self.field_name}: unchanged"
        elif self.diff_type == DiffType.CHANGED:
            return f"{self.field_name}: {self.old_value} → {self.new_value}"
        elif self.diff_type == DiffType.ADDED:
            return f"{self.field_name}: added ({self.new_value})"
        else:  # REMOVED
            return f"{self.field_name}: removed ({self.old_value})"

class ModelDiff(Generic[T]):
    """Utility for comparing two Pydantic models."""
    
    def __init__(self, existing_model: T, new_model: T, exclude_fields: Set[str] = None):
        self.existing_model = existing_model
        self.new_model = new_model
        self.exclude_fields = exclude_fields or {"id"}  # Default to excluding ID
        self._diffs = None
    
    @property
    def diffs(self) -> Dict[str, FieldDiff]:
        """Get field differences between the models, caching the result."""
        if self._diffs is None:
            self._diffs = self._compute_diffs()
        return self._diffs
    
    def _compute_diffs(self) -> Dict[str, FieldDiff]:
        """Compute differences between the models."""
        dict1 = self.existing_model.dict(exclude=self.exclude_fields)
        dict2 = self.new_model.dict(exclude=self.exclude_fields)
        
        all_fields = set(dict1.keys()) | set(dict2.keys())
        result = {}
        
        for field in all_fields:
            if field in self.exclude_fields:
                continue
                
            if field not in dict1:
                result[field] = FieldDiff(field, None, dict2[field], DiffType.ADDED)
            elif field not in dict2:
                result[field] = FieldDiff(field, dict1[field], None, DiffType.REMOVED)
            elif dict1[field] != dict2[field]:
                result[field] = FieldDiff(field, dict1[field], dict2[field], DiffType.CHANGED)
            else:
                result[field] = FieldDiff(field, dict1[field], dict2[field], DiffType.UNCHANGED)
                
        return result
    
    @property
    def has_changes(self) -> bool:
        """Check if there are any differences between the models."""
        return any(diff.diff_type != DiffType.UNCHANGED for diff in self.diffs.values())
    
    def get_changes_summary(self) -> str:
        """Get a human-readable summary of the changes."""
        if not self.has_changes:
            return "No changes"
            
        changed_fields = [
            diff for diff in self.diffs.values() 
            if diff.diff_type != DiffType.UNCHANGED
        ]
        
        return "\n".join(str(diff) for diff in changed_fields)
    
    def to_table(self) -> "Table":
        """Convert the diff to a rich Table for display."""
        from rich.table import Table
        
        table = Table(title=f"Model Comparison")
        
        table.add_column("Field", style="cyan")
        table.add_column("Existing Value", style="green")
        table.add_column("New Value", style="yellow")
        table.add_column("Status", style="magenta")
        
        for field_name, diff in self.diffs.items():
            status_style = {
                DiffType.UNCHANGED: "[green]Unchanged[/]",
                DiffType.CHANGED: "[yellow]Changed[/]",
                DiffType.ADDED: "[green]Added[/]",
                DiffType.REMOVED: "[red]Removed[/]"
            }[diff.diff_type]
            
            old_value = str(diff.old_value) if diff.old_value is not None else "N/A"
            new_value = str(diff.new_value) if diff.new_value is not None else "N/A"
            
            table.add_row(
                field_name.capitalize(),
                old_value,
                new_value,
                status_style
            )
            
        return table