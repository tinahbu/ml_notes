
Convert to PDF by: 
```bash
pandoc --toc overfitting_regularization.md --output overfitting_regularization.pdf
# --toc: generates table of contents

# or
pandoc overfitting_regularization.md --output overfitting_regularization.pdf --from markdown --template eisvogel --listings --toc --toc-own-page true --highlight-style tango
# list of highlight styles: 
# pygments
# tango
# espresso
# zenburn
# kate
# monochrome
# breezedark
# haddock

```
