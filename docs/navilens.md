## NaviLens

NaviLens is a service that provides navigational data resolution, based on ddTags ("distant dense tag")
placed around the environment. The tags are designed so that they can be quickly scanned
with a mobile device while moving or on moving objects and at a much greater distance than
QR codes. A scanned tag is converted into a message on a users device, the message is then sent
to the NaviLens web service, which responds with the associated data for the tag.

The main use case of NaviLens is to improve navigation for those with visual impairments.
For example a tag could be placed on the front of a bus so that a visually impaired person
can hold their phone camera up to the approaching bus to read the route number.

These tag images must be requested from NaviLens who are the central data resolution authority and
maintain a database with the associated data for each ddTag.

### Unique Tags

The typical 5x5 ddTag can only represent 16,777,216 unique combinations. The first line of defence against
running out of codes is to use larger dimension codes (see specification below).

I speculate that another avenue that NaviLens will explore is to divide the world into smaller geographic
regions, within which it allocates tags to users. This allows tags to be safely re-used without collision.