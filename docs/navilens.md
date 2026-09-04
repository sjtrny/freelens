# NaviLens

[NaviLens](https://www.navilens.com/) is an accessible wayfinding system that uses
[ddTags](./ddtag.md), or "distant dense tags." These small grids of coloured cells are
designed to be detected at greater distances, with less precise camera aim, and while
the tag or user is in relative motion. That makes them useful on signs, stops, vehicles,
and packaging, especially when a person cannot first see and centre a marker in the
camera.

## Why ddTags?

QR codes can contain URLs or other useful data directly, but their dense grid must be
framed closely and resolved clearly. Scanning one on a moving vehicle is difficult, and
finding one in the first place can be a barrier for people with visual or motor
impairments.

ddTags trade capacity for easier detection. Each cell has one of four colours and
therefore represents two bits. After reserving cells for orientation and error
detection, the common 5x5 tag carries a 24-bit message. Its simple, low-density pattern
is what allows a reader to locate it under less controlled conditions. See
[Detecting ddTags](./ddtag-detection.md) for the process used by FreeLens.

## How NaviLens works

The 24-bit message is only an identifier. NaviLens resolves that identifier using a
hosted database:

![NaviLens system overview: an app detects a ddTag ID, exchanges it with the NaviLens service over the Internet, and receives information from the tag registry](./assets/navilens/system-overview.svg)

The app scans the camera image, extracts the tag ID, and sends it to the NaviLens
service. The service looks up the information registered for that ID and returns it for
the app to present. This lets an operator update or localise the information without
printing a new tag. A tag on a tram, for example, can resolve to route and stop
information that the app reads aloud.

This separation is the main difference between a ddTag and the complete NaviLens system.
A ddTag decoder can recover the number locally; it cannot know what that number means
without NaviLens's registry or another source containing the same mapping.

## A physical namespace

The registry acts as a physical namespace: numbered tags stand for places, objects, or
signs in much the same way that a domain name stands for an Internet destination. A 5x5
tag has 2^24, or 16,777,216, possible messages, so IDs must be allocated consistently to
prevent two deployments assigning different meanings to the same tag. Larger ddTags
provide more IDs, but do not remove the need for a registry in this model.

![Conceptual 24-bit tag namespace divided into general-purpose, personal-use, and commercial ranges, with commercial leases](./assets/navilens/namespace.svg)

One possible way to stretch the limited ID space would be to divide the world into
geographic regions and reuse the same ID in places far enough apart that they cannot be
confused. The service could then resolve the combination of tag ID and location.

Central resolution makes small tags capable of returning rich, changeable information,
but it also creates a dependency on the registry operator. Access to registered tags,
service availability, pricing, long-term stewardship, and the metadata exposed by remote
lookups all matter when the system becomes public accessibility infrastructure. A QR
code that embeds its content directly does not have the same service dependency.

## FreeLens

FreeLens is not affiliated with NaviLens. It is an understandable reference
implementation for generating and detecting ddTags, and a starting point for experiments
with open alternatives. It does not provide NaviLens's official tag registry or its
registered content. The community image dataset helps test detection under the varied
lighting, devices, distances, and perspectives that real wayfinding requires. Its
sources and licence are recorded in the [dataset notes](../dataset/DATASET.md).
