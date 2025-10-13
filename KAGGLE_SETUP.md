# Kaggle API Setup Guide

This guide walks you through setting up the Kaggle API to download datasets for the Diabetic Retinopathy classification project.

## Overview

The Kaggle API allows programmatic access to Kaggle competitions and datasets. This project uses it to automatically download the APTOS 2019 Blindness Detection dataset.

## Prerequisites

- Python 3.8 or higher
- Active Kaggle account (free to create at [kaggle.com](https://www.kaggle.com))
- Internet connection

## Installation Steps

### 1. Install the Kaggle Package

The `kaggle` package is included in `requirements.txt`. Install it with:

```bash
pip install kaggle
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
kaggle --version
```

You should see something like: `Kaggle API 1.5.x`

### 2. Obtain Your Kaggle API Credentials

1. **Log in to Kaggle:**
   - Go to [kaggle.com](https://www.kaggle.com) and sign in

2. **Access Your Account Settings:**
   - Click on your profile picture (top right)
   - Select "Settings" from the dropdown menu
   - Or go directly to: [kaggle.com/settings](https://www.kaggle.com/settings/account)

3. **Create API Token:**
   - Scroll down to the "API" section
   - Click the **"Create New API Token"** button
   - This will download a file named `kaggle.json` to your computer

4. **Important:** The `kaggle.json` file contains:
   ```json
   {
     "username": "your_kaggle_username",
     "key": "your_api_key_here"
   }
   ```
   Keep this file secure! It grants access to your Kaggle account.

### 3. Place the Credentials File

The Kaggle API looks for credentials in a specific location:

**On macOS/Linux:**

```bash
# Create the .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set appropriate permissions (REQUIRED for security)
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows:**

```powershell
# Create the .kaggle directory in your user folder
mkdir C:\Users\<YourUsername>\.kaggle

# Move the downloaded kaggle.json file
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# Windows handles permissions differently, but ensure the file is not publicly readable
```

**Expected file location:**
- **macOS/Linux:** `~/.kaggle/kaggle.json`
- **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### 4. Set File Permissions (macOS/Linux only)

This is a **critical security step** to protect your API credentials:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

This command ensures:
- Only you (the file owner) can read and write the file
- No other users on the system can access it

**Common permission error:**
If you see: `"OSError: Could not find kaggle.json. Make sure it's located in ~/.kaggle..."`, check:
1. The file exists in the correct location
2. The file has correct permissions (600)
3. The file contains valid JSON

## Verification

### Quick Test

Run the test script to verify your setup:

```bash
python scripts/test_kaggle.py
```

This will check:
- ✓ Kaggle package is installed
- ✓ kaggle.json exists and is readable
- ✓ File has correct permissions
- ✓ API authentication works
- ✓ Can access Kaggle competitions

### Manual Verification

You can also test manually:

```bash
# List your Kaggle competitions (should not error)
kaggle competitions list

# Check if you can access the APTOS competition
kaggle competitions list | grep aptos
```

If these commands work without errors, your setup is complete!

## Using the Kaggle API

### Download APTOS Dataset

Once configured, you can download datasets:

```bash
# Using the project's data preparation script
python scripts/prepare_data.py --aptos-only

# Or manually with kaggle CLI
kaggle competitions download -c aptos2019-blindness-detection
```

### Accept Competition Rules

**Important:** Before downloading competition data, you must:
1. Visit the competition page: [kaggle.com/c/aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
2. Click "Join Competition" or "I Understand and Accept"
3. Accept the competition rules

If you skip this step, you'll get an error: `"403 - Forbidden"`

## Troubleshooting

### Error: "Could not find kaggle.json"

**Solution:**
- Verify the file is at `~/.kaggle/kaggle.json` (macOS/Linux) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
- Check the path with: `ls -la ~/.kaggle/` (macOS/Linux)

### Error: "Permission denied" or "OSError: Could not read kaggle.json"

**Solution (macOS/Linux):**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

**Solution (Windows):**
- Right-click `kaggle.json` → Properties → Security
- Ensure only your user account has read permissions

### Error: "401 - Unauthorized"

**Causes:**
- Invalid API credentials
- Expired or regenerated token

**Solution:**
1. Delete the old token: [kaggle.com/settings](https://www.kaggle.com/settings/account)
2. Generate a new API token
3. Replace `~/.kaggle/kaggle.json` with the new file
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Error: "403 - Forbidden"

**Cause:** You haven't accepted the competition rules

**Solution:**
1. Visit: [kaggle.com/c/aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
2. Click "Join Competition"
3. Accept the rules
4. Try downloading again

### Error: "404 - Not Found"

**Causes:**
- Competition name is incorrect
- Competition is no longer available

**Solution:**
- Verify the competition exists at kaggle.com
- Check for typos in the competition name

### Slow Download Speed

**Solutions:**
- Use a wired internet connection
- Download during off-peak hours
- Use the `-q` flag for quiet mode: `kaggle competitions download -q -c aptos2019-blindness-detection`

## Advanced Configuration

### Custom Credentials Location

You can specify a custom location for `kaggle.json`:

```bash
export KAGGLE_CONFIG_DIR=/path/to/custom/location
```

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) to make it permanent.

### Proxy Configuration

If you're behind a proxy:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=https://proxy.example.com:8080
```

### Environment Variables

Instead of `kaggle.json`, you can use environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**Note:** Using `kaggle.json` is more secure and recommended.

## Security Best Practices

1. **Never commit kaggle.json to version control**
   - The `.gitignore` file should exclude `kaggle.json`
   - Never share your API key publicly

2. **Regenerate tokens periodically**
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings/account)
   - Click "Create New API Token" (this invalidates the old one)

3. **Use restrictive file permissions**
   - Always set `chmod 600` on `kaggle.json` (macOS/Linux)

4. **Monitor API usage**
   - Check your Kaggle account for unexpected activity
   - Regenerate token if compromised

## Additional Resources

- **Official Kaggle API Documentation:** [github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api)
- **Kaggle API Reference:** [kaggle.com/docs/api](https://www.kaggle.com/docs/api)
- **APTOS Competition:** [kaggle.com/c/aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **Kaggle Account Settings:** [kaggle.com/settings](https://www.kaggle.com/settings/account)

## Quick Reference

| Task | Command |
|------|---------|
| Install Kaggle | `pip install kaggle` |
| Check version | `kaggle --version` |
| Test authentication | `kaggle competitions list` |
| Download APTOS dataset | `python scripts/prepare_data.py --aptos-only` |
| Test project setup | `python scripts/test_kaggle.py` |
| Set permissions | `chmod 600 ~/.kaggle/kaggle.json` |
| View competitions | `kaggle competitions list` |
| Download competition files | `kaggle competitions download -c <competition-name>` |

## Getting Help

If you encounter issues:

1. **Run the test script:** `python scripts/test_kaggle.py`
2. **Check the error message** and refer to the Troubleshooting section above
3. **Consult the official docs:** [github.com/Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api)
4. **Check Kaggle forums:** [kaggle.com/discussions](https://www.kaggle.com/discussions)

## Next Steps

Once your Kaggle API is configured:

1. ✓ Test the setup: `python scripts/test_kaggle.py`
2. ✓ Accept APTOS competition rules: [kaggle.com/c/aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
3. ✓ Download the dataset: `python scripts/prepare_data.py --aptos-only`
4. ✓ Begin training: `python scripts/train_retfound_lora.py`

---

**Last Updated:** 2025-10-13
