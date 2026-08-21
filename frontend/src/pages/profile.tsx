import React, { useState } from 'react';
import { Edit3 } from 'lucide-react';
import { PageHeader, Card, Button, Field, SelectField, Notice, StatusPill } from '@/components/gatesense-ui';
import { useAuth } from '@/lib/auth';

interface ProfileFormData {
  fullName: string;
  email: string;
  username: string;
  accountStatus: string;
}

export default function ProfilePage() {
  const { user, updateUser, isLoading } = useAuth();

  // Dynamic user data from authentication session
  const currentName = user?.name || 'Operator';
  const currentEmail = user?.email || 'operator@anprx.io';
  const currentUsername = user?.username || user?.email?.split('@')[0] || 'operator';
  const currentStatus = user?.accountStatus || 'Active';

  // State for editing profile fields: Full Name, Email Address, Username, Account Status
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState<ProfileFormData>({
    fullName: currentName,
    email: currentEmail,
    username: currentUsername,
    accountStatus: currentStatus,
  });
  const [profileNotice, setProfileNotice] = useState<string | null>(null);

  const handleStartEdit = () => {
    setFormData({
      fullName: user?.name || currentName,
      email: user?.email || currentEmail,
      username: user?.username || currentUsername,
      accountStatus: user?.accountStatus || currentStatus,
    });
    setIsEditing(true);
    setProfileNotice(null);
  };

  const handleCancelEdit = () => {
    setFormData({
      fullName: user?.name || currentName,
      email: user?.email || currentEmail,
      username: user?.username || currentUsername,
      accountStatus: user?.accountStatus || currentStatus,
    });
    setIsEditing(false);
  };

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await updateUser({
        name: formData.fullName.trim(),
        email: formData.email.trim().toLowerCase(),
        username: formData.username.trim(),
        accountStatus: formData.accountStatus,
      });
      setIsEditing(false);
      setProfileNotice('Profile updated successfully.');
      setTimeout(() => {
        setProfileNotice(null);
      }, 4000);
    } catch (err: any) {
      setProfileNotice('Failed to update profile: ' + (err.message || 'Error'));
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Account / Profile"
        title="Profile"
        description="View and manage your account details."
      />

      {profileNotice && (
        <div className="mb-6 animate-slide-in">
          <Notice kind={profileNotice.startsWith('Failed') ? 'bad' : 'good'}>
            {profileNotice}
          </Notice>
        </div>
      )}

      <div className="max-w-3xl">
        <Card
          title="Account Profile"
          action={
            !isEditing ? (
              <Button
                variant="secondary"
                onClick={handleStartEdit}
                testId="button-edit-profile"
                className="text-xs py-1.5"
              >
                <Edit3 className="h-3.5 w-3.5" />
                Edit Profile
              </Button>
            ) : null
          }
        >
          <div className="p-6">
            {!isEditing ? (
              <div className="grid gap-6 sm:grid-cols-2">
                <InfoItem label="Full Name" value={currentName} />
                <InfoItem label="Email Address" value={currentEmail} />
                <InfoItem label="Username" value={currentUsername.startsWith('@') ? currentUsername : `@${currentUsername}`} />
                <div className="rounded-lg border border-border/70 bg-background/35 p-3.5">
                  <p className="text-[10px] font-bold uppercase tracking-[.14em] text-muted-foreground">
                    Account Status
                  </p>
                  <div className="mt-2">
                    <StatusPill value={currentStatus} />
                  </div>
                </div>
              </div>
            ) : (
              <form onSubmit={handleSaveProfile} className="space-y-5">
                <div className="grid gap-5 sm:grid-cols-2">
                  <Field
                    label="Full Name"
                    value={formData.fullName}
                    onChange={(v) => setFormData({ ...formData, fullName: v })}
                    placeholder="Enter full name"
                    required
                  />
                  <Field
                    label="Email Address"
                    type="email"
                    value={formData.email}
                    onChange={(v) => setFormData({ ...formData, email: v })}
                    placeholder="Enter email address"
                    required
                  />
                  <Field
                    label="Username"
                    value={formData.username}
                    onChange={(v) => setFormData({ ...formData, username: v.replace(/^@/, '') })}
                    placeholder="Enter username"
                    required
                  />
                  <SelectField
                    label="Account Status"
                    value={formData.accountStatus}
                    onChange={(v) => setFormData({ ...formData, accountStatus: v })}
                    options={['Active', 'Inactive', 'Suspended', 'Pending']}
                  />
                </div>

                <div className="flex justify-end gap-3 pt-5 border-t border-border/60">
                  <Button
                    variant="ghost"
                    onClick={handleCancelEdit}
                    testId="button-cancel-edit"
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    variant="primary"
                    disabled={isLoading}
                    testId="button-save-profile"
                  >
                    Save Changes
                  </Button>
                </div>
              </form>
            )}
          </div>
        </Card>
      </div>
    </>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border/70 bg-background/35 p-3.5">
      <p className="text-[10px] font-bold uppercase tracking-[.14em] text-muted-foreground">
        {label}
      </p>
      <p className="data-text mt-1 text-sm font-semibold text-foreground">
        {value || '—'}
      </p>
    </div>
  );
}
