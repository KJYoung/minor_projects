import { BootstrapDialog, BootstrapDialogTitle } from "../general/Dialog";
import { styled } from "styled-components";
import { Button, DialogActions, DialogContent } from "@mui/material";
import { workbook, xlsxParser } from "../../utils/FileInterface";
import { useState } from "react";
import { DEFAULT_OPTION } from "../../utils/Constants";

interface TrxnExcelDialogProps {
    open: boolean,
    handleClose: () => void,
    workbook: workbook,
};

export const TrxnExcelDialog = ({open, handleClose, workbook} : TrxnExcelDialogProps) => {
    const workbookSheetNames = workbook.SheetNames;
    const [sheetSelect, setSheetSelect] = useState<string>(DEFAULT_OPTION);
    const SheetContent = () => {
        if(sheetSelect !== DEFAULT_OPTION){
            const parsedSheet = xlsxParser(workbook.Sheets[sheetSelect]);
            if(parsedSheet.length > 0){
                const keys = Object.keys(parsedSheet[0] as any);
                console.log(parsedSheet);
                console.log(keys);
                return <>
                    {parsedSheet.map((row: any) => {
                        return <>
                            {keys.map((key) => {
                                return <span key={key}>{row[key] && row[key]}, </span>
                            })}
                            <br />
                        </>
                    })}
                </> 
            }
        }
        return <></>
    }
    return <div>
      <BootstrapDialog
        onClose={handleClose}
        aria-labelledby="customized-dialog-title"
        open={open}
        fullWidth={true}
        maxWidth={false}
      >
        <DialogBody>
          <BootstrapDialogTitle id="customized-dialog-title" onClose={handleClose}>
            태그 설정
            <select value={sheetSelect} onChange={(e) => setSheetSelect(e.target.value)}>
                <option disabled value={DEFAULT_OPTION}>
                    - 시트를 선택하세요. -
                </option>
                {workbookSheetNames.map((sheetName) => <option value={sheetName} key={sheetName}>{sheetName}</option>)}
            </select>
          </BootstrapDialogTitle>
          <DialogContent dividers> 
            {sheetSelect !== DEFAULT_OPTION && <SheetContent />}  
          </DialogContent>
          <DialogActions>
            <Button autoFocus onClick={handleClose}>
              닫기
            </Button>
          </DialogActions>
        </DialogBody>
      </BootstrapDialog>
    </div>
};

const DialogBody = styled.div`
  width: 100%;
`;