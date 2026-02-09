Add a new strata filter to the Streamlit sidebar for a specified column.

## Arguments

$ARGUMENTS: The column name to add as a filter (e.g., `sub_grade`, `application_type`)

## Steps

1. Read `.claude/skills/data-schema/SKILL.md` to verify the column exists and understand its values
2. Read `.claude/skills/streamlit-patterns/SKILL.md` for the filter pattern

3. In `app.py`, add a new multiselect widget in the sidebar section, following the existing pattern:

```python
# [Column Name] filter
all_values = sorted(df['COLUMN_NAME'].dropna().unique())
selected_values = st.multiselect(
    "Label",
    options=all_values,
    default=all_values,
    key="COLUMN_NAME_filter"
)
```

4. Add the filter to the filtering logic:

```python
df_filtered = df[
    ... &  # existing filters
    (df['COLUMN_NAME'].isin(selected_values))
]
```

5. Verify the filter works:
   - Launch the app (`streamlit run app.py`)
   - Confirm the new filter appears in the sidebar
   - Confirm selecting/deselecting values updates the loan count
   - Confirm it doesn't break existing filters or any tab

## Important

- Place the new filter in a logical position among existing filters (e.g., `sub_grade` after `grade`)
- Always default to "Select All" so the filter is non-restrictive on first load
- Use a descriptive label (e.g., "Application Type" not "application_type")
- Add the column to the strata filter list in `.claude/skills/data-schema/SKILL.md` if not already there
- Test that Tab 2 and Tab 3 still work after adding the filter (empty cohort handling)
