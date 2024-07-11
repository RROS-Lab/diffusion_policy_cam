def delete_child_items(item):
    # Retrieve all child items inside the parent item
    child_items = item.Childs()

    # Iterate through each child item and delete it
    for child in child_items:
        child.Delete()

    print(f"All child items inside '{item.Name()}' have been deleted.")


