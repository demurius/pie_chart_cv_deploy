function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('Email Extraction')
    .addItem('Sync Emails', 'triggerEmailSync')
    .addToUi();
}

function triggerEmailSync() {
  const url = 'https://piechart.up.railway.app/sync-emails';
  const apiToken = 'a3fc748973aa0699231e79e69506cc08c6f674f170a9fecb24ba4c509b16b673';
  
  const options = {
    'method': 'get',
    'headers': {
      'Authorization': 'Bearer ' + apiToken
    },
    'muteHttpExceptions': true
  };

  try {
    const response = UrlFetchApp.fetch(url, options);
    const responseCode = response.getResponseCode();
    
    //if (responseCode === 200) {
    //  SpreadsheetApp.getUi().alert('Sync Successful!');
    //} else {
    //  SpreadsheetApp.getUi().alert('Sync failed with status: ' + responseCode);
    //}
  } catch (e) {
    SpreadsheetApp.getUi().alert('Error syncing emails: ' + e.toString());
  }
}
