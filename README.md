This script will download photos from flashphotography.com
It works by simulating clicking the various locations on the blurry photo and stitching together the tiny high-resolution thumbnails.

To run it, you must install these libraries:
  pip install shutil math random requests numpy collections PIL

There are no command-line options.

The script will ask you for three pieces of information.
  1. The URL for the photos. It's the URL after you click a photo and it goes to the order page.
     example (non-functional): https://orders.flashphotography.com/Orders/Packages.aspx?O=27191641&R=00002&F=1891&ViewID=63414&GroupImage=

  2. The ID for that URL, provided in the email from flashphotography (example from email -> "ID: 11706057").
     example: 11706057

  3. The last name of the user associated with the URL and ID
     example: jones

It will then attemnpt to load the page, identify all the photos, and generate final version into an "output" directory.

Note that landscape photos don't work correctly, but most I've seen are portrait.

I have successfully run this on 3 different accounts, including one 3 years old.
