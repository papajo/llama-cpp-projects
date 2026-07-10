"""
Code generation / completion prompts.

These have structured, predictable patterns that n-gram strategies
should handle well (due to repeated keywords, brackets, and syntax).
"""

CODE_PROMPTS = [
    (
        "code-bubble-sort",
        """Write a Python function that implements bubble sort. Include type hints and a docstring.

```python
def bubble_sort(arr: list[int]) -> list[int]:
    """
"""
    ),
    (
        "code-merge-sort",
        """Implement merge sort in Python with proper type annotations:

```python
def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left: list[int], right: list[int]) -> list[int]:
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

Now also implement quicksort:
"""
    ),
    (
        "code-json-parser",
        """Write a recursive descent JSON parser in Python:

```python
import re
from typing import Any, Dict, List, Union


class JSONParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
    
    def skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
    
    def parse(self) -> Any:
        self.skip_whitespace()
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of input")
        
        char = self.text[self.pos]
        if char == '{':
            return self.parse_object()
        elif char == '[':
            return self.parse_array()
        elif char == '"':
            return self.parse_string()
        elif char == 't' or char == 'f':
            return self.parse_boolean()
        elif char == 'n':
            return self.parse_null()
        else:
            return self.parse_number()
    
    def parse_object(self) -> Dict[str, Any]:
        result = {}
        self.pos += 1  # skip {
        self.skip_whitespace()
"""
    ),
    (
        "code-react-component",
        """Create a React TypeScript component for a data table with sorting, filtering, and pagination:

```typescript
import React, { useState, useMemo } from 'react';

interface Column<T> {
    key: keyof T;
    header: string;
    sortable?: boolean;
    filterable?: boolean;
    render?: (value: T[keyof T], row: T) => React.ReactNode;
}

interface DataTableProps<T> {
    data: T[];
    columns: Column<T>[];
    pageSize?: number;
    initialSort?: { key: keyof T; direction: 'asc' | 'desc' };
}

function DataTable<T extends Record<string, any>>({
    data,
    columns,
    pageSize = 10,
    initialSort,
}: DataTableProps<T>) {
    const [sortKey, setSortKey] = useState<keyof T | null>(
        initialSort?.key ?? null
    );
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>(
        initialSort?.direction ?? 'asc'
    );
    const [filters, setFilters] = useState<Partial<Record<keyof T, string>>>({});
    const [page, setPage] = useState(1);

    // Filtered data
    const filteredData = useMemo(() => {
        return data.filter(row => {
            return columns.every(col => {
                const filter = filters[col.key];
                if (!filter) return true;
                const value = String(row[col.key]);
                return value.toLowerCase().includes(filter.toLowerCase());
            });
        });
    }, [data, filters, columns]);

    // Sorted data
    const sortedData = useMemo(() => {
        if (!sortKey) return filteredData;
        return [...filteredData].sort((a, b) => {
            const aVal = a[sortKey];
            const bVal = b[sortKey];
"""
    ),
    (
        "code-api-endpoints",
        """Design a FastAPI backend with the following endpoints for a task management system:

```python
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from datetime import datetime

app = FastAPI(title="Task Manager API")

# Models
class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    priority: int = 0
    assignee: Optional[str] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[int] = None
    assignee: Optional[str] = None

class Task(TaskCreate):
    id: int
    status: TaskStatus = TaskStatus.TODO
    created_at: datetime
    updated_at: datetime

# In-memory store
tasks: List[Task] = []
next_id: int = 1

@app.get("/tasks", response_model=List[Task])
async def list_tasks(
    status: Optional[TaskStatus] = Query(None),
    priority_min: Optional[int] = Query(None),
    assignee: Optional[str] = Query(None),
    sort_by: str = Query("created_at"),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
):
    """List tasks with filtering, sorting, and pagination."""
    result = tasks.copy()
    
    # Filter
    if status:
        result = [t for t in result if t.status == status]
    if priority_min is not None:
        result = [t for t in result if t.priority >= priority_min]
    if assignee:
        result = [t for t in result if t.assignee == assignee]
"""
    ),
]
