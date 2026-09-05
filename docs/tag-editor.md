# Tag editor

Use the tag editor to review labels and edit tag locations in the field photographs. Refer to [Dataset](./dataset.md) for the label fields, benchmark, and photograph licence.

## Start the editor

Run the commands on this page from the repository root.

To start the editor locally, run these commands:

```bash
python -m pip install -e ".[editor]"
python -m scripts.tag_editor
```

As an alternative, use this command to build and start the service in Docker:

```bash
docker compose up --build
```

Open [the tag editor](http://localhost:8899) on the Docker host. From a different machine, use the host's address with port 8899. Compose publishes this port on all host interfaces.

The editor has no user authentication. Make it available only on a trusted network.

## Select and edit a tag

The image shows tags with known locations as green outlines. A selected tag has a red outline and four corner handles. Edits also update the square, perspective-corrected preview below the tag details.

1. Click a green tag to select it.
1. Drag its corner handles to fit the tag boundary.
1. Click **Save** to store the changes.

To clear the selection, click a part of the image with no tag. If there are unsaved changes, the editor asks how to handle them before a selection change. The choices are to save, discard, or continue the edit. Click **Cancel** to restore the last saved values.

## Add a tag or bounding box

1. Click **Add tag** adjacent to the tag-list heading.
1. Adjust the new bounding box to fit the tag.
1. If the message is known, enter it in the message field. If not, keep the field empty.
1. Click **Save**.

For an existing tag without a location, click **Add bounding box**. The new box is a square at the centre of the image region in view. Its size depends on the current zoom.

## Move and resize the view

- To move a whole box, drag within it.
- To scale a box about its opposite corner, hold **Shift** as you start to drag a corner.
- To pan a magnified image, drag outside the boxes.
- To zoom about the pointer, scroll over the image. There is no upper zoom limit.

## Data storage and Docker settings

The editor checks each saved change before it replaces `dataset/evaluation.json`. This file replacement is atomic. Compose permits these writes through the `./dataset:/app/dataset` bind mount. All other application paths are read-only.

Compose runs the editor with UID and GID 1000 by default. On Linux, change these values if a different user owns the checkout. If Docker runs separately from the development container, use a checkout path that the Docker daemon can access. Set `TAG_EDITOR_DATASET_PATH` to that path.

You can put these values in the ignored `.env` file:

```bash
TAG_EDITOR_DATASET_PATH=/daemon/path/to/freelens/dataset
TAG_EDITOR_UID=1000
TAG_EDITOR_GID=1000
```

To stop the service, run this command:

```bash
docker compose down
```
